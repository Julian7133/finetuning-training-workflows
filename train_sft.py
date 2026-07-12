"""Build the LoRA training data split and run `mlx_lm.lora` SFT training.

Consumes the data pipeline's judge results + filtered replay pool from the
sibling `Finetuning-data-workflows/llm-usage-extract` repo (Plan v2 Stage 3 /
Stage 5c output) directly, rather than the already-merged `sft_mixed.jsonl` --
that file drops `record_id`, which we need to hold out a domain-episode eval
split before merging in replay rows. Runs identically for both orchestrator
modes (sft_only / staged_dpo); the eval split is always written, even in
sft_only mode, so a later staged_dpo run can reuse it without retraining.

completion_status (Plan v2) was never implemented in the judge schema, so the
eval split is a random 10% of domain episodes (not stratified by
nudge/abandoned status -- see build_nudge_flags.py docstring for why that
proxy has too few hits to stratify on). preceded_by_stall_nudge is attached
as metadata only, when available.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# Configured oMLX memory ceiling on the target M3 is tighter than the full
# 24GB unified memory pool (soft=17.8GB, hard=19.9GB, ceil=21.0GB) -- a prior
# run that pushed the ceiling higher froze the system. Canary runs compare
# observed peak RSS against these, not the theoretical 24GB.
OMLX_SOFT_LIMIT_GB = 17.8
OMLX_HARD_LIMIT_GB = 19.9
OMLX_CEILING_GB = 21.0

from chatml import render_chatml

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_ROOT = Path(__file__).resolve().parent.parent / "Finetuning-data-workflows" / "llm-usage-extract"
DEFAULT_MODEL = "andjiang/CoPaw-Flash-9B-oQ4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SFT+replay train/eval split and run mlx_lm.lora."
    )
    parser.add_argument(
        "--pipeline-root",
        default=str(DEFAULT_PIPELINE_ROOT),
        help="Path to Finetuning-data-workflows/llm-usage-extract (source of judge_results.jsonl etc.)",
    )
    parser.add_argument("--domain-input", default=None, help="Override judge_results.jsonl path")
    parser.add_argument("--replay-input", default=None, help="Override replay_filtered.jsonl path")
    parser.add_argument("--nudge-flags", default=None, help="Override nudge_flags.jsonl path")
    parser.add_argument("--coherence-floor", type=int, default=3)
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.12,
        help="Target replay share of the merged training set (matches build_sft.py Stage 5d default)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.10,
        help="Fraction of accepted domain episodes to hold out as eval (random split)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
        help="Examples tokenizing longer than this are dropped (not truncated) before training. "
        "32768 covers 95.8pct of domain episodes (553/577); the 24 dropped are the true "
        "outlier tail (p99=78455, max=142392). 16384 would cover only 88.7pct and "
        "disproportionately drops the highest-tool-call-density trajectories (~3.8x the "
        "average tool-call count) -- exactly the examples this fine-tune cares most about.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=-1,
        help="Number of transformer blocks (from the end) to LoRA-adapt; -1 = all 32. "
        "mlx_lm's own default is 16 (last half only). -1 was our initial choice reading "
        "the plan's 'all-linear' as full-depth too, which combined with grad_checkpoint "
        "may be what triggered a Metal resource-count crash during canary testing.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action="store_true",
        default=True,
        help="Gradient checkpointing (default on). Trades peak memory for more live Metal "
        "resources per step (recompute graphs) -- implicated in the canary resource-limit crash.",
    )
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="LoRA alpha; converted to mlx_lm's `scale = alpha / rank` internally",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0, help="Startwert, not fixed -- see Plan v4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8, help="8-16 target effective batch size at batch-size=1")
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=8,
        help="Validation batches per eval (-1 = full eval set). Full-set eval at these "
        "sequence lengths costs ~1-3min per example -- with ~58 eval rows that is more "
        "than an hour per eval, every steps_per_eval iterations. 8 batches keeps a "
        "usable signal at a fraction of the cost.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Checkpoint frequency. Lowered from mlx_lm's default 100 -- the observed crashes "
        "are an accumulating Metal resource-count effect, not a hard limit at a fixed config, "
        "so a mid-run crash is a realistic scenario and losing <25 iterations matters more "
        "than the extra checkpoint I/O overhead.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest numbered checkpoint in --adapter-path, if one exists. "
        "mlx_lm only restores weights (not optimizer/Adam state), so this is a crash-recovery "
        "mechanism, not a bitwise-identical continuation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="output/mlx_data")
    parser.add_argument("--adapter-path", default="output/adapters")
    parser.add_argument("--config-out", default="output/lora_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Build the split + config, skip mlx_lm.lora")
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="Run only a short burst (see --canary-iters) with RSS memory polling, report peak "
        "memory against the oMLX ceiling (soft 17.8GB/hard 19.9GB/ceil 21.0GB), then exit "
        "without running the full training. Use this before a full run at a new max_seq_length.",
    )
    parser.add_argument(
        "--canary-iters",
        type=int,
        default=20,
        help="Number of training iterations for --canary-only",
    )
    parser.add_argument(
        "--canary-poll-seconds",
        type=float,
        default=1.0,
        help="RSS sampling interval during --canary-only",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR"])
    return parser.parse_args(argv)


@dataclass(frozen=True)
class SplitStats:
    domain_accepted: int
    domain_train: int
    domain_eval: int
    replay_available: int
    replay_selected: int
    train_written: int
    eval_written: int
    train_dropped_too_long: int
    eval_dropped_too_long: int


def load_pipeline_modules(pipeline_root: Path):
    if not pipeline_root.is_dir():
        raise FileNotFoundError(
            f"Pipeline root not found: {pipeline_root}. Pass --pipeline-root explicitly."
        )
    sys.path.insert(0, str(pipeline_root))
    import build_sft  # noqa: PLC0415
    import schema  # noqa: PLC0415

    return build_sft, schema


def load_nudge_flags(path: Path) -> dict[str, bool]:
    if not path.exists():
        logger.warning(
            "No nudge-flags file at %s -- preceded_by_stall_nudge metadata will be false "
            "for all rows. Run build_nudge_flags.py first if you want it populated.",
            path,
        )
        return {}
    flags: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            flags[obj["record_id"]] = obj["preceded_by_stall_nudge"]
    return flags


def episode_messages_to_dicts(episode) -> list[dict]:
    return [
        {"role": msg.role.value, "content": msg.content, "tool_name": msg.tool_name}
        for msg in episode.messages
    ]


def build_domain_rows(build_sft, domain_input: Path, coherence_floor: int, nudge_flags: dict[str, bool]) -> list[dict]:
    rows: list[dict] = []
    for line, system_prompt in build_sft.read_judge_results(domain_input):
        if build_sft.rejection_reason(line.judge, coherence_floor) is not None:
            continue
        messages = episode_messages_to_dicts(line.episode)
        rows.append(
            {
                "record_id": line.record_id,
                "text": render_chatml(messages, system_prompt=system_prompt),
                "source": "domain",
                "preceded_by_stall_nudge": nudge_flags.get(line.record_id, False),
            }
        )
    return rows


def build_replay_rows(replay_rows: list[dict]) -> list[dict]:
    rendered: list[dict] = []
    for row in replay_rows:
        messages = row.get("messages")
        if not messages:
            messages = [
                {"role": "user", "content": row.get("prompt", "")},
                {"role": "assistant", "content": row.get("completion", "")},
            ]
        entry = {"text": render_chatml(messages), "source": "replay"}
        if "category" in row:
            entry["category"] = row["category"]
        rendered.append(entry)
    return rendered


def split_domain_rows(
    rows: list[dict], eval_fraction: float, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    if not 0 <= eval_fraction < 1:
        raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_count = round(eval_fraction * len(shuffled))
    if eval_fraction > 0 and eval_count == 0 and shuffled:
        eval_count = 1
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def filter_by_length(rows: list[dict], tokenizer, max_seq_length: int) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        if token_length(tokenizer, row["text"]) <= max_seq_length:
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


@dataclass(frozen=True)
class MergeResult:
    rows: list[dict]
    replay_available: int
    replay_selected: int


def merge_domain_and_replay(
    build_sft, domain_rows: list[dict], raw_replay_rows: list[dict], target_ratio: float, rng: random.Random
) -> MergeResult:
    target_replay_count = build_sft.compute_target_replay_count(len(domain_rows), target_ratio)
    selected_raw = build_sft.sample_balanced_replay(raw_replay_rows, target_replay_count, rng)
    replay_rows = build_replay_rows(selected_raw)
    mixed = domain_rows + replay_rows
    rng.shuffle(mixed)
    return MergeResult(rows=mixed, replay_available=len(raw_replay_rows), replay_selected=len(replay_rows))


def write_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_lora_config(args: argparse.Namespace, data_dir: Path, iters: int) -> dict:
    return {
        "model": args.model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "seed": args.seed,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": iters,
        "val_batches": args.val_batches,
        "learning_rate": args.learning_rate,
        "steps_per_report": args.steps_per_report,
        "steps_per_eval": args.steps_per_eval,
        "adapter_path": args.adapter_path,
        "save_every": args.save_every,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": args.grad_checkpoint,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "lora_parameters": {
            "rank": args.rank,
            "scale": args.alpha / args.rank,
            "dropout": args.dropout,
        },
    }


def find_latest_checkpoint(adapter_path: Path) -> tuple[Path, int] | None:
    """mlx_lm saves numbered snapshots as `{iter:07d}_adapters.safetensors` every
    save_every iterations (in addition to a rolling `adapters.safetensors`). Only
    weights are saved, not optimizer state -- resuming restarts Adam momentum."""
    if not adapter_path.is_dir():
        return None
    checkpoints = []
    for path in adapter_path.glob("*_adapters.safetensors"):
        try:
            iteration = int(path.stem.split("_")[0])
        except ValueError:
            continue
        checkpoints.append((path, iteration))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[1])


TRAIN_RUNNER = Path(__file__).resolve().parent / "train_runner.py"


def _runner_cmd(config_path: Path) -> list[str]:
    """train_runner.py wraps mlx_lm.lora with the memory patches this model
    needs on 24GB (time-chunked GatedDeltaNet backward, chunked 248k-vocab CE,
    clamped wired limit). Calling `mlx_lm lora` directly reproduces the
    ~21GB+ first-backward peak regardless of max_seq_length."""
    return [sys.executable, str(TRAIN_RUNNER), "-c", str(config_path)]


def run_with_memory_guard(cmd: list[str], poll_seconds: float = 1.0) -> tuple[int, float | None, bool]:
    """Run cmd while polling phys_footprint_peak; kill at the oMLX ceiling
    (raising past it froze the system before). Returns (returncode,
    peak_gb_observed, killed_by_guard)."""
    proc = subprocess.Popen(cmd)
    samples: list[float] = []
    killed = threading.Event()
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            peak = read_phys_footprint_peak_gb(proc.pid)
            if peak is not None:
                samples.append(peak)
                if peak >= OMLX_CEILING_GB:
                    logger.error(
                        "Memory guard: %.1fGB >= ceiling %.1fGB -- killing training process.",
                        peak,
                        OMLX_CEILING_GB,
                    )
                    killed.set()
                    proc.kill()
                    return
            stop.wait(poll_seconds)

    poller = threading.Thread(target=watch)
    poller.start()
    returncode = proc.wait()
    stop.set()
    poller.join()
    return returncode, (max(samples) if samples else None), killed.is_set()


def run_training(config_path: Path) -> int:
    cmd = _runner_cmd(config_path)
    logger.info("Running: %s", " ".join(cmd))
    returncode, peak_gb, killed = run_with_memory_guard(cmd)
    if peak_gb is not None:
        logger.info("Training peak phys_footprint: %.2f GB", peak_gb)
    if killed:
        logger.error(
            "Training was killed by the memory guard at the %.1fGB ceiling. "
            "Checkpoints up to the last save_every interval are in the adapter path; "
            "rerun with --resume after lowering max_seq_length.",
            OMLX_CEILING_GB,
        )
        return 75  # distinct exit code for guard kill
    return returncode


def read_phys_footprint_peak_gb(pid: int) -> float | None:
    """`ps -o rss=` drastically undercounts MLX/Metal workloads on Apple Silicon --
    GPU-resident (IOAccelerator) allocations don't show up in classic Mach VM RSS.
    `footprint <pid>`'s `phys_footprint_peak` is the number that actually tracks
    against the unified-memory ceiling (confirmed empirically: ps reported ~5GB
    RSS while footprint reported an 18GB peak for the same process at the same
    moment)."""
    result = subprocess.run(["footprint", str(pid)], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("phys_footprint_peak:"):
            value = line.split(":", 1)[1].strip()
            if value.endswith("GB"):
                return float(value[:-2].strip())
            if value.endswith("MB"):
                return float(value[:-2].strip()) / 1024
    return None


def poll_peak_footprint_gb(pid: int, interval_seconds: float, stop: threading.Event) -> list[float]:
    samples: list[float] = []
    while not stop.is_set():
        peak = read_phys_footprint_peak_gb(pid)
        if peak is not None:
            samples.append(peak)
        stop.wait(interval_seconds)
    return samples


def longest_rows(rows: list[dict], tokenizer, count: int) -> list[dict]:
    """The canary should probe worst-case memory, not an average sample -- with
    batch_size=1, iterate_batches visits examples in random order, so a plain
    20-iteration slice would likely miss the long tail we actually care about."""
    scored = sorted(rows, key=lambda r: token_length(tokenizer, r["text"]), reverse=True)
    return scored[:count]


def run_canary(
    config_path: Path, canary_rows: list[dict], canary_iters: int, poll_seconds: float
) -> float | None:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    canary_data_dir = Path(config["data"]) / "_canary"
    write_jsonl(canary_rows, canary_data_dir / "train.jsonl")
    write_jsonl(canary_rows, canary_data_dir / "valid.jsonl")

    config["data"] = str(canary_data_dir)
    config["iters"] = min(canary_iters, len(canary_rows))
    config["adapter_path"] = str(Path(config["adapter_path"]) / "_canary")
    config["save_every"] = canary_iters + 1  # don't bother checkpointing a throwaway run
    # val_batches=-1 (the real config's setting) evaluates the ENTIRE valid set --
    # here that's all `canary_rows` again, which turns a "quick 20-iter probe"
    # into 20 more near-max-length forward passes on top of training. Cap it.
    config["val_batches"] = min(3, len(canary_rows))

    canary_config_path = config_path.with_name(config_path.stem + "_canary.yaml")
    with canary_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    cmd = _runner_cmd(canary_config_path)
    logger.info("Running canary (%d iters): %s", canary_iters, " ".join(cmd))
    returncode, peak_gb, killed = run_with_memory_guard(cmd, poll_seconds)

    if peak_gb is None:
        logger.warning("No footprint samples collected during canary run")
        return None
    if returncode != 0 and not killed:
        logger.error("Canary run exited with code %d -- see mlx_lm output above", returncode)
        return None
    return peak_gb


def report_canary_result(peak_gb: float, canary_row_count: int, shortest_canary_tokens: int) -> None:
    print(
        f"Canary ran the {canary_row_count} longest training examples "
        f"(shortest of them: {shortest_canary_tokens} tokens) -- a worst-case-length probe, "
        f"not a representative-average sample."
    )
    print(f"Canary peak RSS: {peak_gb:.2f} GB")
    print(
        f"oMLX ceiling on this M3: soft={OMLX_SOFT_LIMIT_GB}GB "
        f"hard={OMLX_HARD_LIMIT_GB}GB ceil={OMLX_CEILING_GB}GB"
    )
    if peak_gb >= OMLX_HARD_LIMIT_GB:
        print(
            "OVER the hard limit -- do not run the full training at this max_seq_length. "
            "Fall back to a lower value (e.g. 24576, 93.6pct of domain episodes kept) and "
            "re-run the canary."
        )
    elif peak_gb >= OMLX_SOFT_LIMIT_GB:
        print(
            "Above the soft limit but under hard/ceil. This is close enough to the limit that "
            "was previously followed by a system freeze when raised -- recommend a lower "
            "max_seq_length (e.g. 24576) rather than proceeding at 32768."
        )
    else:
        print("Under the soft limit on the worst-case examples. Still your call -- report this back before the full run.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline_root = Path(args.pipeline_root)
    build_sft, _schema = load_pipeline_modules(pipeline_root)

    domain_input = Path(args.domain_input) if args.domain_input else pipeline_root / "output" / "judge_results.jsonl"
    replay_input = Path(args.replay_input) if args.replay_input else pipeline_root / "output" / "replay_filtered.jsonl"
    nudge_flags_path = Path(args.nudge_flags) if args.nudge_flags else pipeline_root / "output" / "nudge_flags.jsonl"

    nudge_flags = load_nudge_flags(nudge_flags_path)
    domain_rows = build_domain_rows(build_sft, domain_input, args.coherence_floor, nudge_flags)
    if not domain_rows:
        logger.error("No accepted domain episodes found in %s", domain_input)
        return 1

    rng = random.Random(args.seed)
    train_domain, eval_domain = split_domain_rows(domain_rows, args.eval_fraction, rng)

    raw_replay_rows = build_sft.read_jsonl(replay_input)
    merge_result = merge_domain_and_replay(
        build_sft, train_domain, raw_replay_rows, args.target_ratio, rng
    )
    mixed_train_rows = merge_result.rows

    from mlx_lm.utils import load_tokenizer  # noqa: PLC0415

    tokenizer = load_tokenizer(args.model)

    train_rows, train_dropped = filter_by_length(mixed_train_rows, tokenizer, args.max_seq_length)
    eval_rows, eval_dropped = filter_by_length(eval_domain, tokenizer, args.max_seq_length)

    if not train_rows:
        logger.error("All training rows were dropped for exceeding max_seq_length=%d", args.max_seq_length)
        return 1
    if not eval_rows:
        logger.warning(
            "All eval rows were dropped for exceeding max_seq_length=%d -- validation will be skipped",
            args.max_seq_length,
        )

    data_dir = Path(args.data_dir)
    train_written = write_jsonl(train_rows, data_dir / "train.jsonl")
    eval_written = write_jsonl(eval_rows, data_dir / "valid.jsonl")

    stats = SplitStats(
        domain_accepted=len(domain_rows),
        domain_train=len(train_domain),
        domain_eval=len(eval_domain),
        replay_available=merge_result.replay_available,
        replay_selected=merge_result.replay_selected,
        train_written=train_written,
        eval_written=eval_written,
        train_dropped_too_long=train_dropped,
        eval_dropped_too_long=eval_dropped,
    )
    print(
        f"Domain episodes accepted: {stats.domain_accepted} "
        f"(train {stats.domain_train} / eval {stats.domain_eval}, "
        f"random split, seed={args.seed})"
    )
    print(
        f"Replay merged: {stats.replay_selected}/{stats.replay_available} "
        f"(target ratio {args.target_ratio})"
    )
    print(
        f"Dropped for exceeding max_seq_length={args.max_seq_length}: "
        f"{stats.train_dropped_too_long} train, {stats.eval_dropped_too_long} eval"
    )
    print(f"Wrote {stats.train_written} train rows, {stats.eval_written} eval rows to {data_dir}")

    iters = max(1, round(args.epochs * len(train_rows) / args.batch_size))
    config = build_lora_config(args, data_dir, iters)

    if args.resume:
        checkpoint = find_latest_checkpoint(Path(args.adapter_path))
        if checkpoint is None:
            logger.warning("--resume requested but no checkpoint found in %s -- starting fresh", args.adapter_path)
        else:
            checkpoint_path, completed_iters = checkpoint
            remaining = iters - completed_iters
            if remaining <= 0:
                logger.error(
                    "Checkpoint %s already covers the target iters (%d >= %d) -- nothing to resume",
                    checkpoint_path,
                    completed_iters,
                    iters,
                )
                return 1
            config["resume_adapter_file"] = str(checkpoint_path)
            config["iters"] = remaining
            print(
                f"Resuming from {checkpoint_path} ({completed_iters} iters already done, "
                f"{remaining} remaining -- note: optimizer/Adam state is not restored, only weights)"
            )

    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote LoRA config to {config_path} (iters={config['iters']}, ~{args.epochs} epochs)")

    if args.dry_run:
        print("Dry run: skipping mlx_lm.lora invocation")
        return 0

    if args.canary_only:
        canary_rows = longest_rows(train_rows, tokenizer, args.canary_iters)
        shortest_canary_tokens = token_length(tokenizer, canary_rows[-1]["text"])
        peak_gb = run_canary(config_path, canary_rows, args.canary_iters, args.canary_poll_seconds)
        if peak_gb is None:
            logger.error("Canary run did not complete -- see errors above")
            return 1
        report_canary_result(peak_gb, len(canary_rows), shortest_canary_tokens)
        print("Canary-only: not running the full training. Re-run without --canary-only once satisfied.")
        return 0

    return run_training(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
