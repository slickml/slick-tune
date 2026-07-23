# Warm-start with LoRA SFT, then GRPO with verifiable rewards, then probe.
# GRPO alone on a cold 135M base almost never hits exact ``must_contain``
# phrases, so group advantages stay zero (loss/grad = 0). SFT first.
# Usage from repo root:
#   uv run python examples/run_grpo_lora.py

from __future__ import annotations

from pathlib import Path

from slicktune import GRPOObjective, LoRAStrategy, SFTObjective, Tuner
from slicktune.recipes import load_trained, run_probes

ROOT = Path(__file__).resolve().parents[1]
SFT_DATA = ROOT / "examples" / "data" / "about_amir.jsonl"
GRPO_DATA = ROOT / "examples" / "data" / "about_amir.grpo.jsonl"
PROBES = ROOT / "examples" / "data" / "about_amir.probes.jsonl"
SFT_OUT = ROOT / "outputs" / "grpo_lora_sft"
OUT = ROOT / "outputs" / "grpo_lora"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> int:
    """Run SFT warm-start → LoRA GRPO and print probe pass rate."""
    strategy = LoRAStrategy(r=16, alpha=32)
    print("=== SFT warm-start ===")
    Tuner(
        model_id=MODEL_ID,
        strategy=strategy,
        objective=SFTObjective(),
        output_dir=SFT_OUT,
        num_train_epochs=15,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_seq_length=512,
    ).fit(SFT_DATA)

    print("=== GRPO ===")
    result = Tuner(
        model_id=MODEL_ID,
        strategy=strategy,
        objective=GRPOObjective(
            num_generations=4,
            max_completion_length=96,
            temperature=0.9,
            beta=0.0,
        ),
        output_dir=OUT,
        adapter_path=SFT_OUT,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_seq_length=512,
        probe_path=PROBES,
    ).fit(GRPO_DATA)

    model, tokenizer = load_trained(result.output_dir)
    report = run_probes(model=model, tokenizer=tokenizer, probe_path=PROBES)
    print(f"Saved to {result.output_dir}")
    print(f"train_loss={result.metrics.train_loss}")
    if result.metrics.judge_score is not None:
        print(f"judge_score={result.metrics.judge_score:.0%}")
    print(f"probe_pass_rate={report.pass_rate:.0%}")
    for item in report.results:
        mark = "PASS" if item.passed else "FAIL"
        print(f"[{mark}] {item.prompt!r} -> {item.generation!r}")

    return 0 if report.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
