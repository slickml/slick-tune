# Train QLoRA SFT (CUDA + bitsandbytes required).
#   uv sync --extra qlora
#   uv run python examples/run_sft_qlora.py
#
# On Apple Silicon / CPU, use LoRA instead:
#   uv run python examples/run_sft_lora.py

from __future__ import annotations

import sys
from pathlib import Path

import torch

from slicktune import QLoRAStrategy, SFTObjective, Tuner
from slicktune.recipes import load_trained, run_probes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "about_amir.jsonl"
PROBES = ROOT / "examples" / "data" / "about_amir.probes.jsonl"
OUT = ROOT / "outputs" / "sft_qlora"


def main() -> None:
    """Run a small QLoRA SFT job and print probe pass rate."""
    if not torch.cuda.is_available():
        print(
            "QLoRA needs a CUDA GPU (bitsandbytes 4-bit).\n"
            "On Apple Silicon / CPU run:\n"
            "  uv run python examples/run_sft_lora.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    result = Tuner(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        strategy=QLoRAStrategy(r=8, alpha=16),
        objective=SFTObjective(),
        output_dir=OUT,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=512,
    ).fit(DATA)

    model, tokenizer = load_trained(result.output_dir)
    report = run_probes(model, tokenizer, PROBES)
    print(f"Saved to {result.output_dir}")
    print(f"train_loss={result.metrics.train_loss}")
    print(f"probe_pass_rate={report.pass_rate:.0%}")
    for item in report.results:
        mark = "PASS" if item.passed else "FAIL"
        print(f"[{mark}] {item.prompt!r} -> {item.generation!r}")


if __name__ == "__main__":
    main()
