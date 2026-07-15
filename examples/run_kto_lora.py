# Train LoRA KTO on unpaired preference labels, then probe.
# Usage from repo root:
#   uv run python examples/run_kto_lora.py

from __future__ import annotations

from pathlib import Path

from slicktune import KTOObjective, LoRAStrategy, Tuner
from slicktune.recipes import load_trained, run_probes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "about_amir.kto.jsonl"
PROBES = ROOT / "examples" / "data" / "about_amir.probes.jsonl"
EVAL = ROOT / "examples" / "data" / "about_amir.eval.jsonl"
OUT = ROOT / "outputs" / "kto_lora"


def main() -> int:
    """Run a small LoRA KTO job and print probe pass rate."""
    result = Tuner(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        strategy=LoRAStrategy(r=16, alpha=32),
        objective=KTOObjective(beta=0.1),
        output_dir=OUT,
        eval_data=EVAL,
        num_train_epochs=10,
        # TRL KTO requires per-device batch size > 1 for a meaningful KL term.
        per_device_train_batch_size=2,
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
