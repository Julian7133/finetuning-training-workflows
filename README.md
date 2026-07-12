# finetuning-training-workflows

LoRA SFT for [`andjiang/CoPaw-Flash-9B-oQ4`](https://huggingface.co/andjiang/CoPaw-Flash-9B-oQ4) on domain + replay data from the sibling `Finetuning-data-workflows/llm-usage-extract` pipeline.

## Quick start

```bash
pip install -r requirements.txt

# Build split + canary (Option B: mlx-lm-lora @ 4096, default backend)
python train_sft.py --canary-only --canary-iters 3 --max-seq-length 4096 --num-layers 16

# Full CoPaw SFT @ 4096 (~862 iters, grad_checkpoint + efficient_long_context)
python train_sft.py --max-seq-length 4096 --num-layers 16

# Long-context worst-case (8192+) — manual backend only on 24GB M3
python train_sft.py --backend manual --canary-only --canary-iters 3 --max-seq-length 8192
python train_sft.py --backend manual --max-seq-length 8192 --num-layers 16
```

## mlx-lm-lora DPO reproduction (reference notebook)

Reproduces [Qwen3_4B_Gabliterated_DPO.ipynb](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks/blob/main/preference/Qwen3_4B_Gabliterated_DPO.ipynb) with the same model (`Goekdeniz-Guelmez/Qwen3-4B-Instruct-2507-gabliterated`) at `max_seq_length=4096`:

```bash
python reproduce_dpo_qwen3.py --max-seq-length 4096 --skip-merge
# Smoke test (10 samples): ~8 min, completes successfully
# Full notebook parity (100 samples): ~90 min
```

## Backends

| Backend | Engine | When to use |
|---|---|---|
| `mlx-lm-lora` (default) | [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) via `mlx_lora_runner.py` | **Option B:** `@ 4096` with `grad_checkpoint=True` — canary peak **11 GB** on worst-case examples. Default backend. |
| `manual` | `train_runner.py` | Required for 8192-token worst-case on 24GB M3 (GatedDeltaNet manual BPTT). Slower (~90s/iter) but memory-safe. |

See [`finetuning_runbook.ipynb`](finetuning_runbook.ipynb) for the full runbook.
