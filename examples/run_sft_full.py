# Full fine-tuning baseline on the tiny personal dataset (memory-heavy).
# Prefer LoRA for day-to-day iteration.
#   uv run python examples/run_sft_full.py

from __future__ import annotations

from pathlib import Path

from slicktune import FullStrategy, SFTObjective, Tuner
from slicktune.recipes import load_trained, run_probes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "about_amir.jsonl"
PROBES = ROOT / "examples" / "data" / "about_amir.probes.jsonl"
OUT = ROOT / "outputs" / "sft_full"


def main() -> None:
    """Run a tiny full-FT SFT job and print probe pass rate."""
    result = Tuner(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        strategy=FullStrategy(),
        objective=SFTObjective(),
        output_dir=OUT,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
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
