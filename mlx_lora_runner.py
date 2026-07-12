"""Launch mlx_lm_lora.train with CoPaw-safe memory settings.

Patches applied before training:
- Wired limit clamp (14GB cap; stock trainer can wire ~22GB on this M3)
- seq_step_size from YAML / COPAW_SEQ_STEP_SIZE (stock train.run hardcodes 512)

Usage: python mlx_lora_runner.py -c output/lora_mlx_lora_config.yaml
"""

from __future__ import annotations

import dataclasses
import os
import sys

import mlx.core as mx

GB = 1024**3
WIRED_CAP_BYTES = int(14.0 * GB)
CACHE_LIMIT_BYTES = int(2 * GB)
DEFAULT_SEQ_STEP = int(os.environ.get("COPAW_SEQ_STEP_SIZE", "512"))


def _install_wired_limit_clamp() -> None:
    orig = mx.set_wired_limit

    def capped(limit):
        return orig(min(int(limit), WIRED_CAP_BYTES))

    mx.set_wired_limit = capped


def _seq_step_for(args) -> int:
    if getattr(args, "seq_step_size", None) is not None:
        return int(args.seq_step_size)
    return DEFAULT_SEQ_STEP


def _patch_efficient_seq_step() -> None:
    """mlx-lm-lora train.run() hardcodes seq_step_size=512 for efficient_long_context."""
    import mlx_lm_lora.train as lora_train
    from mlx_lm_lora.trainer import sft_trainer

    orig_run = lora_train.run

    def run(args):
        if getattr(args, "efficient_long_context", False):
            args.seq_step_size = _seq_step_for(args)
        step = _seq_step_for(args) if getattr(args, "efficient_long_context", False) else None

        orig_train_sft = lora_train.train_sft

        def train_sft(*a, **kw):
            sft_args = kw.get("args")
            if step is not None and sft_args is not None:
                kw["args"] = dataclasses.replace(sft_args, seq_step_size=step)
            return orig_train_sft(*a, **kw)

        lora_train.train_sft = train_sft
        sft_trainer.train_sft = train_sft
        try:
            return orig_run(args)
        finally:
            lora_train.train_sft = orig_train_sft
            sft_trainer.train_sft = orig_train_sft

    lora_train.run = run


def main() -> int:
    _install_wired_limit_clamp()
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    _patch_efficient_seq_step()

    from mlx_lm_lora.train import main as lora_main

    lora_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
