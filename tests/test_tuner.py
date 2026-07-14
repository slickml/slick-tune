"""Tests for Tuner validation and mocked fit path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from assertpy import assert_that
from datasets import Dataset

from slicktune import LoRAStrategy, SFTObjective, Tuner
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
    monkeypatch.setattr("slicktune.tuner.load_model", lambda model_id, strategy: model)
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
    monkeypatch.setattr("slicktune.tuner.load_model", lambda model_id, strategy: model)
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
