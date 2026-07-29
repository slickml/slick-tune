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

## ✅ LoRA + GRPO (verifiable rewards)

GRPO JSONL uses `prompt` + `must_contain` (same idea as probes). Completions that
contain the substring get reward `1.0`. On a cold tiny base, warm-start from SFT
first via `Tuner.adapter_path`. The CLI `train` command has no `--adapter-path`,
so use the smoke example:

```bash
# SFT warm-start → GRPO (outputs/grpo_lora_sft then outputs/grpo_lora)
poe train-grpo
# or: uv run python examples/run_grpo_lora.py
```

## ✅ Merge adapters (TIES / DARE)

Smoke demo trains two tiny adapters then merges them:

```bash
poe merge-ties
# or: uv run python examples/run_merge_ties.py
# → outputs/merge_a_lora + outputs/merge_b_lora → outputs/merged_ties
```

```bash
uv run slicktune merge \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter outputs/merge_a_lora \
  --adapter outputs/merge_b_lora:0.5 \
  --method ties \
  --density 0.5 \
  --output outputs/merged_ties

# bake into full weights: add --bake
# alternate: merge any two trained adapters, e.g. outputs/sft_lora + outputs/dpo_lora:0.5
```

Details: [Fine-Tuning Guide §14](fine_tuning_guide.md#multi-adapter-merge-ties-dare).

## ✅ Python API

```python
from slicktune import AdapterRef, LoRAStrategy, SFTObjective, Tuner, merge_adapters

Tuner(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    strategy=LoRAStrategy(r=16, alpha=32),
    objective=SFTObjective(),
    output_dir="outputs/sft_lora",
    eval_data="examples/data/about_amir.eval.jsonl",
).fit("examples/data/about_amir.jsonl")

merge_adapters(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    adapters=[
        AdapterRef(path="outputs/merge_a_lora", name="a", weight=1.0),
        AdapterRef(path="outputs/merge_b_lora", name="b", weight=0.5),
    ],
    output_dir="outputs/merged_ties",
    method="ties",
    density=0.5,
)
```

## ✅ Other strategies

```bash
uv run python examples/run_sft_dora.py
uv run python examples/run_sft_adalora.py
uv run python examples/run_sft_full.py
# CUDA + bitsandbytes:
uv sync --extra qlora && uv run python examples/run_sft_qlora.py
```

New to adapters, objectives, or TIES/DARE? Read the {doc}`Fine-Tuning Guide <fine_tuning_guide>`.
