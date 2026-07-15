"""Tests for model and tokenizer loading helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from assertpy import assert_that

from slicktune.models import load_model, load_tokenizer, resolve_dtype
from slicktune.strategies import LoRAStrategy


def test_resolve_dtype_float32_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default dtype is float32 when CUDA is unavailable."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert_that(resolve_dtype()).is_equal_to(torch.float32)


def test_resolve_dtype_bfloat16_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer bfloat16 when CUDA supports it."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)
    assert_that(resolve_dtype()).is_equal_to(torch.bfloat16)


def test_resolve_dtype_float32_when_bf16_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to float32 when CUDA lacks bf16."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)
    assert_that(resolve_dtype()).is_equal_to(torch.float32)


def test_load_tokenizer_sets_pad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing pad_token is filled from eos_token."""
    tok = SimpleNamespace(pad_token=None, eos_token="<eos>", padding_side="left")
    monkeypatch.setattr(
        "slicktune.models.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tok,
    )
    out = load_tokenizer("fake-model")
    assert_that(out.pad_token).is_equal_to("<eos>")
    assert_that(out.padding_side).is_equal_to("right")


def test_load_tokenizer_keeps_existing_pad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing pad_token is preserved."""
    tok = SimpleNamespace(pad_token="<pad>", eos_token="<eos>", padding_side="left")
    monkeypatch.setattr(
        "slicktune.models.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tok,
    )
    out = load_tokenizer("fake-model")
    assert_that(out.pad_token).is_equal_to("<pad>")


def test_load_model_mps_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move model to MPS when available and no device_map."""
    model = MagicMock()
    model.to.return_value = model
    monkeypatch.setattr(
        "slicktune.models.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr("slicktune.models.resolve_dtype", lambda: torch.float32)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    out = load_model(model_id="fake-model", strategy=LoRAStrategy())
    assert_that(out).is_equal_to(model)
    model.to.assert_called_once_with("mps")


def test_load_model_cuda_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use device_map=auto on CUDA without quantization."""
    captured: dict[str, object] = {}

    def _from_pretrained(model_id: str, **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "slicktune.models.AutoModelForCausalLM.from_pretrained",
        _from_pretrained,
    )
    monkeypatch.setattr("slicktune.models.resolve_dtype", lambda: torch.float32)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    load_model(model_id="fake-model", strategy=LoRAStrategy())
    assert_that(captured).contains_key("device_map")
    assert_that(captured["device_map"]).is_equal_to("auto")


def test_load_model_cuda_without_device_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to .to(cuda) when CUDA is up but MPS is not."""
    model = MagicMock()
    model.to.return_value = model

    class _Strategy:
        def load_kwargs(self) -> dict[str, object]:
            return {"quantization_config": object()}

    monkeypatch.setattr(
        "slicktune.models.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    out = load_model(model_id="fake-model", strategy=_Strategy())  # type: ignore[arg-type]
    assert_that(out).is_equal_to(model)
    model.to.assert_called_once_with("cuda")


def test_load_model_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave model on CPU when neither MPS nor CUDA is available."""
    model = MagicMock()
    monkeypatch.setattr(
        "slicktune.models.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr("slicktune.models.resolve_dtype", lambda: torch.float32)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    out = load_model(model_id="fake-model", strategy=LoRAStrategy())
    assert_that(out).is_equal_to(model)
    model.to.assert_not_called()
