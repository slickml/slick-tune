📌 Quick Start
================

Default demo model: `HuggingFaceTB/SmolLM2-135M-Instruct` (small enough for laptop smoke tests).

## ✅ LoRA + SFT + probes

```bash
uv run slicktune train \
  --strategy lora \
  --data examples/data/about_amir.jsonl \
  --eval-data examples/data/about_amir.eval.jsonl \
  --output outputs/sft_lora \
  --epochs 20

uv run slicktune probe \
  --model-dir outputs/sft_lora \
  --probes examples/data/about_amir.probes.jsonl

uv run slicktune eval \
  --model-dir outputs/sft_lora \
  --eval-data examples/data/about_amir.eval.jsonl \
  --probes examples/data/about_amir.probes.jsonl \
  --judge substring
```

Or with Poe / examples:

```bash
poe train-lora
poe probe-lora
poe eval-lora
# or
uv run python examples/run_sft_lora.py
```

## ✅ LoRA + DPO (preference pairs)

```bash
uv run slicktune train \
  --objective dpo \
  --strategy lora \
  --data examples/data/about_amir.prefs.jsonl \
  --eval-data examples/data/about_amir.eval.jsonl \
  --output outputs/dpo_lora \
  --epochs 10 \
  --beta 0.1

# or
poe train-dpo
```

## ✅ LoRA + KTO (unpaired labels)

```bash
uv run slicktune train \
  --objective kto \
  --strategy lora \
  --data examples/data/about_amir.kto.jsonl \
  --eval-data examples/data/about_amir.eval.jsonl \
  --output outputs/kto_lora \
  --epochs 10 \
  --beta 0.1

# or
poe train-kto
```

ORPO uses `--objective orpo` with the same preference JSONL as DPO (TRL experimental trainer).

## ✅ Python API

```python
from slicktune import LoRAStrategy, SFTObjective, Tuner

Tuner(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    strategy=LoRAStrategy(r=16, alpha=32),
    objective=SFTObjective(),
    output_dir="outputs/sft_lora",
    eval_data="examples/data/about_amir.eval.jsonl",
).fit("examples/data/about_amir.jsonl")
```

## ✅ Other strategies

```bash
uv run python examples/run_sft_dora.py
uv run python examples/run_sft_adalora.py
uv run python examples/run_sft_full.py
# CUDA + bitsandbytes:
uv sync --extra qlora && uv run python examples/run_sft_qlora.py
```

New to adapters? Read the {doc}`Fine-Tuning Guide <fine_tuning_guide>`.
