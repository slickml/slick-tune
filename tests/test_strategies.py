"""Tests for strategy configuration and apply paths."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from assertpy import assert_that

from slicktune.strategies import FullStrategy, LoRAStrategy, QLoRAStrategy


def test_lora_strategy_defaults() -> None:
    """LoRA exposes expected PEFT defaults."""
    strategy = LoRAStrategy()
    assert_that(strategy.name).is_equal_to("lora")
    assert_that(strategy.r).is_equal_to(16)
    assert_that(strategy.load_kwargs()).is_equal_to({})


def test_lora_strategy_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """LoRA apply builds a LoraConfig and wraps the model."""
    fake_peft = MagicMock(name="peft_model")
    monkeypatch.setattr("slicktune.strategies.get_peft_model", lambda model, cfg: fake_peft)
    monkeypatch.setattr(
        "slicktune.strategies.LoraConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    out = LoRAStrategy(r=8).apply(MagicMock())
    assert_that(out).is_equal_to(fake_peft)


def test_full_strategy_name_and_apply() -> None:
    """Full FT enables gradients on all parameters."""
    strategy = FullStrategy()
    assert_that(strategy.name).is_equal_to("full")
    assert_that(strategy.load_kwargs()).is_equal_to({})

    p1 = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    p2 = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    model = SimpleNamespace(parameters=lambda: iter([p1, p2]))
    out = strategy.apply(model)
    assert_that(out).is_equal_to(model)
    assert_that(p1.requires_grad).is_true()
    assert_that(p2.requires_grad).is_true()


def test_qlora_load_kwargs_behavior() -> None:
    """QLoRA either returns a 4-bit config (CUDA) or raises clearly."""
    strategy = QLoRAStrategy()
    assert_that(strategy.name).is_equal_to("qlora")
    try:
        kwargs = strategy.load_kwargs()
    except (ImportError, RuntimeError):
        return
    assert_that(kwargs).contains_key("quantization_config")


def test_qlora_load_kwargs_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA raises when CUDA is unavailable."""
    monkeypatch.setitem(sys.modules, "bitsandbytes", MagicMock())
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        QLoRAStrategy().load_kwargs()


def test_qlora_load_kwargs_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA returns quantization_config when CUDA + bitsandbytes are available."""
    monkeypatch.setitem(sys.modules, "bitsandbytes", MagicMock())
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    class _FakeBnB:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    with patch("transformers.BitsAndBytesConfig", _FakeBnB):
        kwargs = QLoRAStrategy().load_kwargs()
    assert_that(kwargs).contains_key("quantization_config")
    assert_that(kwargs["quantization_config"].kwargs["load_in_4bit"]).is_true()


def test_qlora_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """QLoRA prepare + LoRA wrap path."""
    prepared = MagicMock(name="prepared")
    wrapped = MagicMock(name="wrapped")
    monkeypatch.setattr(
        "slicktune.strategies.prepare_model_for_kbit_training",
        lambda model: prepared,
    )
    monkeypatch.setattr("slicktune.strategies.get_peft_model", lambda model, cfg: wrapped)
    monkeypatch.setattr(
        "slicktune.strategies.LoraConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    out = QLoRAStrategy().apply(MagicMock())
    assert_that(out).is_equal_to(wrapped)
