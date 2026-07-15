"""Tests for Tuner validation and mocked fit path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from assertpy import assert_that
from datasets import Dataset

from slicktune import AdaLoRAStrategy, LoRAStrategy, SFTObjective, Tuner
from slicktune.eval import HoldoutEvalResult, JudgeReport, JudgeResult
from slicktune.objectives import DPOObjective
from slicktune.tuner import _as_optional_float


def test_tuner_rejects_unimplemented_objective(tmp_path: Path) -> None:
    """Phase-1 tuner only trains SFT."""
    tuner = Tuner(
        model_id="sshleifer/tiny-gpt2",
        strategy=LoRAStrategy(),
        objective=DPOObjective(),
        output_dir=tmp_path / "out",
    )
    data = Dataset.from_list([{"prompt": "a", "chosen": "b", "rejected": "c"}])
    with pytest.raises(TypeError):
        tuner.fit(data)


def test_tuner_rejects_missing_columns(tmp_path: Path) -> None:
    """SFT requires messages column."""
    tuner = Tuner(
        model_id="sshleifer/tiny-gpt2",
        strategy=LoRAStrategy(),
        objective=SFTObjective(),
        output_dir=tmp_path / "out",
    )
    data = Dataset.from_list([{"text": "hello"}])
    with pytest.raises(ValueError) as exc:
        tuner.fit(data)
    assert_that(str(exc.value)).contains("messages")


def test_as_optional_float() -> None:
    """Cast helper returns None or float."""
    assert_that(_as_optional_float(None)).is_none()
    assert_that(_as_optional_float("1.5")).is_equal_to(1.5)


def test_tuner_fit_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path fit with heavy deps mocked."""
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Yo"}]}\n',
        encoding="utf-8",
    )

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = ["chat-text", ["list", "text"]]
    tokenizer.save_pretrained = MagicMock()

    model = MagicMock()
    train_output = SimpleNamespace(
        metrics={
            "train_loss": 0.5,
            "train_runtime": 1.2,
            "train_samples_per_second": 3.4,
            "epoch": 1.0,
        }
    )
    trainer = MagicMock()
    trainer.train.return_value = train_output
    trainer.model = model
    trainer.save_model = MagicMock()

    monkeypatch.setattr("slicktune.tuner.load_tokenizer", lambda model_id: tokenizer)
    monkeypatch.setattr("slicktune.tuner.load_model", lambda *, model_id, strategy: model)
    monkeypatch.setattr(
        "slicktune.tuner.count_parameters",
        lambda m: (10, 100),
    )
    monkeypatch.setattr("slicktune.tuner.SFTConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("slicktune.tuner.SFTTrainer", lambda **kwargs: trainer)
    monkeypatch.setattr(
        "slicktune.recipes.probe.prepare_model_for_inference",
        lambda m: m,
    )

    strategy = SimpleNamespace(name="lora", apply=lambda m: m)

    tuner = Tuner(
        model_id="fake",
        strategy=strategy,  # type: ignore[arg-type]
        objective=SFTObjective(),
        output_dir=tmp_path / "out",
        num_train_epochs=1,
    )
    result = tuner.fit(data_path)
    assert_that(result.output_dir.exists()).is_true()
    assert_that(result.metrics.train_loss).is_equal_to(0.5)
    assert_that(result.metrics.trainable_params).is_equal_to(10)
    assert_that(result.metrics.extras).contains_key("epoch")
    trainer.train.assert_called_once()
    tokenizer.save_pretrained.assert_called_once()


def test_tuner_fit_in_memory_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit accepts an in-memory Dataset directly."""
    dataset = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]}]
    )
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "chat-text"
    tokenizer.save_pretrained = MagicMock()
    model = MagicMock()
    trainer = MagicMock()
    trainer.train.return_value = SimpleNamespace(metrics={})
    trainer.model = model
    trainer.save_model = MagicMock()

    monkeypatch.setattr("slicktune.tuner.load_tokenizer", lambda model_id: tokenizer)
    monkeypatch.setattr("slicktune.tuner.load_model", lambda *, model_id, strategy: model)
    monkeypatch.setattr("slicktune.tuner.count_parameters", lambda m: (1, 1))
    monkeypatch.setattr("slicktune.tuner.SFTConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("slicktune.tuner.SFTTrainer", lambda **kwargs: trainer)
    monkeypatch.setattr(
        "slicktune.recipes.probe.prepare_model_for_inference",
        lambda m: m,
    )

    strategy = SimpleNamespace(name="lora", apply=lambda m: m)

    result = Tuner(
        model_id="fake",
        strategy=strategy,  # type: ignore[arg-type]
        objective=SFTObjective(),
        output_dir=tmp_path / "out2",
    ).fit(dataset)
    assert_that(result.metrics.train_loss).is_none()


def test_tuner_adalora_and_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AdaLoRA adds callback; eval_data/probes populate metrics."""
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Yo"}]}\n',
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n',
        encoding="utf-8",
    )

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "chat-text"
    tokenizer.save_pretrained = MagicMock()
    model = MagicMock()
    trainer = MagicMock()
    trainer.train.return_value = SimpleNamespace(metrics={"train_loss": 0.2})
    trainer.model = model
    trainer.save_model = MagicMock()
    captured: dict[str, object] = {}

    def _trainer(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return trainer

    monkeypatch.setattr("slicktune.tuner.load_tokenizer", lambda model_id: tokenizer)
    monkeypatch.setattr("slicktune.tuner.load_model", lambda *, model_id, strategy: model)
    monkeypatch.setattr("slicktune.tuner.count_parameters", lambda m: (1, 2))
    monkeypatch.setattr("slicktune.tuner.SFTConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("slicktune.tuner.SFTTrainer", _trainer)
    monkeypatch.setattr("slicktune.recipes.probe.prepare_model_for_inference", lambda m: m)
    monkeypatch.setattr(AdaLoRAStrategy, "apply", lambda self, m: m)
    monkeypatch.setattr(
        "slicktune.tuner.compute_holdout_perplexity",
        lambda *a, **k: HoldoutEvalResult(eval_loss=0.4, perplexity=1.5, num_examples=1),
    )
    monkeypatch.setattr(
        "slicktune.tuner.run_judge_on_probes",
        lambda *a, **k: JudgeReport(
            results=[JudgeResult(prompt="Who?", generation="SlickML", score=1.0, rationale="ok")]
        ),
    )

    strategy = AdaLoRAStrategy(total_step=1000)

    result = Tuner(
        model_id="fake",
        strategy=strategy,
        objective=SFTObjective(),
        output_dir=tmp_path / "out3",
        num_train_epochs=2,
        eval_data=data_path,
        probe_path=probes,
    ).fit(data_path)

    assert_that(result.metrics.eval_perplexity).is_equal_to(1.5)
    assert_that(result.metrics.judge_score).is_equal_to(1.0)
    assert_that(result.metrics.probe_pass_rate).is_equal_to(1.0)
    callbacks = captured.get("callbacks")
    assert_that(callbacks).is_not_none()
    assert_that(len(callbacks)).is_equal_to(1)  # type: ignore[arg-type]


def test_tuner_custom_judge_skips_probe_pass_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-substring judges set judge_score but not probe_pass_rate."""
    from slicktune.eval import Judge, JudgeResult

    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Yo"}]}\n',
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n',
        encoding="utf-8",
    )

    class _AlwaysJudge(Judge):
        def judge(self, *, prompt: str, generation: str, **context: object) -> JudgeResult:
            return JudgeResult(prompt=prompt, generation=generation, score=0.5, rationale="custom")

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "chat-text"
    tokenizer.save_pretrained = MagicMock()
    model = MagicMock()
    trainer = MagicMock()
    trainer.train.return_value = SimpleNamespace(metrics={})
    trainer.model = model
    trainer.save_model = MagicMock()

    monkeypatch.setattr("slicktune.tuner.load_tokenizer", lambda model_id: tokenizer)
    monkeypatch.setattr("slicktune.tuner.load_model", lambda *, model_id, strategy: model)
    monkeypatch.setattr("slicktune.tuner.count_parameters", lambda m: (1, 1))
    monkeypatch.setattr("slicktune.tuner.SFTConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("slicktune.tuner.SFTTrainer", lambda **kwargs: trainer)
    monkeypatch.setattr("slicktune.recipes.probe.prepare_model_for_inference", lambda m: m)
    monkeypatch.setattr(
        "slicktune.tuner.run_judge_on_probes",
        lambda *a, **k: JudgeReport(
            results=[JudgeResult(prompt="Who?", generation="x", score=0.5, rationale="custom")]
        ),
    )

    strategy = SimpleNamespace(name="lora", apply=lambda m: m)
    result = Tuner(
        model_id="fake",
        strategy=strategy,  # type: ignore[arg-type]
        objective=SFTObjective(),
        output_dir=tmp_path / "out4",
        probe_path=probes,
        judge=_AlwaysJudge(),
    ).fit(data_path)
    assert_that(result.metrics.judge_score).is_equal_to(0.5)
    assert_that(result.metrics.probe_pass_rate).is_none()


def test_prepare_strategy_keeps_explicit_total_step() -> None:
    """Explicit AdaLoRA total_step is preserved."""
    tuner = Tuner(
        model_id="fake",
        strategy=AdaLoRAStrategy(total_step=42),
        objective=SFTObjective(),
        output_dir="out",
    )
    prepared = tuner._prepare_strategy(10)
    assert_that(prepared).is_instance_of(AdaLoRAStrategy)
    assert_that(cast(AdaLoRAStrategy, prepared).total_step).is_equal_to(42)


def test_prepare_strategy_estimates_default_total_step() -> None:
    """Default AdaLoRA total_step=1000 is replaced with a run estimate."""
    tuner = Tuner(
        model_id="fake",
        strategy=AdaLoRAStrategy(total_step=1000),
        objective=SFTObjective(),
        output_dir="out",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
    )
    prepared = cast(AdaLoRAStrategy, tuner._prepare_strategy(10))
    assert_that(prepared.total_step).is_equal_to(20)
    assert_that(prepared.tinit).is_equal_to(6)
    assert_that(prepared.tfinal).is_equal_to(3)


def test_prepare_strategy_keeps_explicit_tinit_tfinal() -> None:
    """Explicit AdaLoRA tinit/tfinal are preserved when total_step is default."""
    tuner = Tuner(
        model_id="fake",
        strategy=AdaLoRAStrategy(total_step=1000, tinit=3, tfinal=2),
        objective=SFTObjective(),
        output_dir="out",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
    )
    prepared = cast(AdaLoRAStrategy, tuner._prepare_strategy(10))
    assert_that(prepared.total_step).is_equal_to(20)
    assert_that(prepared.tinit).is_equal_to(3)
    assert_that(prepared.tfinal).is_equal_to(2)


def test_prepare_strategy_passthrough_for_lora() -> None:
    """Non-AdaLoRA strategies are returned unchanged."""
    strategy = LoRAStrategy()
    tuner = Tuner(
        model_id="fake",
        strategy=strategy,
        objective=SFTObjective(),
        output_dir="out",
    )
    assert_that(tuner._prepare_strategy(10)).is_equal_to(strategy)
