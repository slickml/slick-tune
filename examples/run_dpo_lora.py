# Train LoRA DPO on preference pairs, then probe.
# Usage from repo root:
#   uv run python examples/run_dpo_lora.py

from __future__ import annotations

from pathlib import Path

from slicktune import DPOObjective, LoRAStrategy, Tuner
from slicktune.recipes import load_trained, run_probes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "about_amir.prefs.jsonl"
PROBES = ROOT / "examples" / "data" / "about_amir.probes.jsonl"
EVAL = ROOT / "examples" / "data" / "about_amir.eval.jsonl"
OUT = ROOT / "outputs" / "dpo_lora"


def main() -> int:
    """Run a small LoRA DPO job and print probe pass rate."""
    result = Tuner(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        strategy=LoRAStrategy(r=16, alpha=32),
        objective=DPOObjective(beta=0.1),
        output_dir=OUT,
        eval_data=EVAL,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_seq_length=512,
    ).fit(DATA)

    model, tokenizer = load_trained(result.output_dir)
    report = run_probes(model=model, tokenizer=tokenizer, probe_path=PROBES)
    print(f"Saved to {result.output_dir}")
    print(f"train_loss={result.metrics.train_loss}")
    if result.metrics.eval_perplexity is not None:
        print(
            f"eval_loss={result.metrics.eval_loss} "
            f"eval_perplexity={result.metrics.eval_perplexity:.3f}"
        )
    print(f"probe_pass_rate={report.pass_rate:.0%}")
    for item in report.results:
        mark = "PASS" if item.passed else "FAIL"
        print(f"[{mark}] {item.prompt!r} -> {item.generation!r}")

    return 0 if report.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
