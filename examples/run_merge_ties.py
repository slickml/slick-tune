# Train two small LoRA adapters, merge with TIES, then probe.
# Usage from repo root:
#   uv run python examples/run_merge_ties.py

from __future__ import annotations

from pathlib import Path

from slicktune import AdapterRef, LoRAStrategy, SFTObjective, Tuner, merge_adapters
from slicktune.recipes import load_trained, run_probes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "about_amir.jsonl"
EVAL = ROOT / "examples" / "data" / "about_amir.eval.jsonl"
PROBES = ROOT / "examples" / "data" / "about_amir.probes.jsonl"
OUT_A = ROOT / "outputs" / "merge_a_lora"
OUT_B = ROOT / "outputs" / "merge_b_lora"
OUT_MERGED = ROOT / "outputs" / "merged_ties"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> int:
    """Train two LoRA adapters, TIES-merge them, and print probe pass rate."""
    # Same r/alpha required for TIES (and dare_ties / dare_linear / linear).
    strategy_a = LoRAStrategy(r=16, alpha=32)
    strategy_b = LoRAStrategy(r=16, alpha=32)

    Tuner(
        model_id=MODEL_ID,
        strategy=strategy_a,
        objective=SFTObjective(),
        output_dir=OUT_A,
        eval_data=EVAL,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_seq_length=512,
    ).fit(DATA)

    Tuner(
        model_id=MODEL_ID,
        strategy=strategy_b,
        objective=SFTObjective(),
        output_dir=OUT_B,
        eval_data=EVAL,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_seq_length=512,
    ).fit(DATA)

    result = merge_adapters(
        model_id=MODEL_ID,
        adapters=[
            AdapterRef(path=OUT_A, name="merge_a_lora", weight=1.0),
            AdapterRef(path=OUT_B, name="merge_b_lora", weight=1.0),
        ],
        output_dir=OUT_MERGED,
        method="ties",
        density=0.5,
        bake=False,
    )

    model, tokenizer = load_trained(result.output_dir)
    report = run_probes(model=model, tokenizer=tokenizer, probe_path=PROBES)
    print(f"Merged adapter saved to {result.output_dir}")
    print(f"probe_pass_rate={report.pass_rate:.0%}")
    for item in report.results:
        mark = "PASS" if item.passed else "FAIL"
        print(f"[{mark}] {item.prompt!r} -> {item.generation!r}")

    return 0 if report.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
