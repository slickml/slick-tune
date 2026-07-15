"""Tests for probe helpers with mocked models/tokenizers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from assertpy import assert_that

from slicktune.recipes import (
    ProbeReport,
    ProbeResult,
    generate_reply,
    load_trained,
    prepare_model_for_inference,
    run_probes,
)


def test_probe_pass_rate() -> None:
    """Pass rate averages boolean outcomes."""
    report = ProbeReport(
        results=[
            ProbeResult(prompt="q1", must_contain="a", generation="a yes", passed=True),
            ProbeResult(prompt="q2", must_contain="b", generation="nope", passed=False),
        ]
    )
    assert_that(report.pass_rate).is_equal_to(0.5)


def test_probe_pass_rate_empty() -> None:
    """Empty report has zero pass rate."""
    assert_that(ProbeReport(results=[]).pass_rate).is_equal_to(0.0)


def test_prepare_model_for_inference_enables_cache() -> None:
    """Inference prep turns on use_cache and eval mode."""

    class _FakeModel:
        def __init__(self) -> None:
            self.training = True
            self.config = SimpleNamespace(use_cache=False)
            self.checkpointing_disabled = False

        def eval(self) -> None:
            self.training = False

        def gradient_checkpointing_disable(self) -> None:
            self.checkpointing_disabled = True

    model = _FakeModel()
    prepare_model_for_inference(model)
    assert_that(model.training).is_false()
    assert_that(model.checkpointing_disabled).is_true()
    assert_that(model.config.use_cache).is_true()


def test_prepare_model_for_inference_with_base_model() -> None:
    """PEFT-style models also prepare the base model."""

    class _Base:
        def __init__(self) -> None:
            self.config = SimpleNamespace(use_cache=False)
            self.disabled = False

        def gradient_checkpointing_disable(self) -> None:
            self.disabled = True

    class _Peft:
        def __init__(self) -> None:
            self._base = _Base()
            self.config = SimpleNamespace(use_cache=False)

        def eval(self) -> None:
            return None

        def gradient_checkpointing_disable(self) -> None:
            return None

        def get_base_model(self) -> _Base:
            return self._base

    model = _Peft()
    prepare_model_for_inference(model)
    assert_that(model._base.disabled).is_true()
    assert_that(model._base.config.use_cache).is_true()
    assert_that(model.config.use_cache).is_true()


def test_prepare_model_for_inference_base_without_hooks() -> None:
    """Base model without checkpointing/config is still accepted."""

    class _Peft:
        def eval(self) -> None:
            return None

        def gradient_checkpointing_disable(self) -> None:
            return None

        def get_base_model(self) -> object:
            return object()

    model = _Peft()
    out = prepare_model_for_inference(model)
    assert_that(out).is_equal_to(model)


def test_prepare_model_without_optional_hooks() -> None:
    """Models without checkpointing/config hooks still eval()."""

    class _Bare:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

    model = _Bare()
    out = prepare_model_for_inference(model)
    assert_that(out).is_equal_to(model)
    assert_that(model.training).is_false()


def _make_tokenizer(
    *, chat_template: str | None = "tpl", pad_token_id: int | None = 0
) -> MagicMock:
    tok = MagicMock()
    tok.chat_template = chat_template
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = 1
    tok.apply_chat_template.return_value = "<chat>"
    tok.return_value = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tok.decode.return_value = "SlickML founder"
    return tok


def _make_model() -> MagicMock:
    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([param])
    model.generate.return_value = torch.tensor([[10, 11, 12, 20, 21]])
    model.config = SimpleNamespace(use_cache=False)
    return model


def test_generate_reply_with_chat_template() -> None:
    """Generate path using apply_chat_template."""
    text = generate_reply(model=_make_model(), tokenizer=_make_tokenizer(), prompt="Who?")
    assert_that(text).is_equal_to("SlickML founder")


def test_generate_reply_without_chat_template() -> None:
    """Fallback prompt format when chat_template is missing."""
    tok = _make_tokenizer(chat_template=None)
    text = generate_reply(model=_make_model(), tokenizer=tok, prompt="Who?")
    assert_that(text).is_equal_to("SlickML founder")
    tok.assert_called()


def test_generate_reply_pad_falls_back_to_eos() -> None:
    """pad_token_id falls back to eos_token_id."""
    model = _make_model()
    tok = _make_tokenizer(pad_token_id=None)
    generate_reply(model=model, tokenizer=tok, prompt="Who?")
    kwargs = model.generate.call_args.kwargs
    assert_that(kwargs["pad_token_id"]).is_equal_to(1)


def test_generate_reply_list_decode() -> None:
    """List decode results are joined."""
    tok = _make_tokenizer()
    tok.decode.return_value = ["Slick", "ML"]
    text = generate_reply(model=_make_model(), tokenizer=tok, prompt="Who?")
    assert_that(text).is_equal_to("Slick ML")


def test_run_probes(tmp_path: Path) -> None:
    """run_probes aggregates per-question pass/fail."""
    path = tmp_path / "probes.jsonl"
    path.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n{"prompt":"Where?","must_contain":"Mars"}\n',
        encoding="utf-8",
    )
    tok = _make_tokenizer()
    tok.decode.side_effect = ["SlickML founder", "Earth"]
    report = run_probes(model=_make_model(), tokenizer=tok, probe_path=path)
    assert_that(report.pass_rate).is_equal_to(0.5)
    assert_that(report.results[0].passed).is_true()
    assert_that(report.results[1].passed).is_false()


def test_load_trained_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapter checkpoints load via AutoPeftModelForCausalLM."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    tok = SimpleNamespace(pad_token=None, eos_token="<eos>")
    model = MagicMock()
    model.to.return_value = model
    model.config = SimpleNamespace(use_cache=False)

    monkeypatch.setattr(
        "slicktune.recipes.probe.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tok,
    )
    fake_peft = MagicMock()
    fake_peft.from_pretrained.return_value = model
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        SimpleNamespace(AutoPeftModelForCausalLM=fake_peft),
    )
    # Re-import path uses `from peft import AutoPeftModelForCausalLM` inside function
    monkeypatch.setattr(
        "peft.AutoPeftModelForCausalLM",
        fake_peft,
        raising=False,
    )
    monkeypatch.setattr("slicktune.recipes.probe.resolve_dtype", lambda: torch.float32)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    loaded_model, loaded_tok = load_trained(tmp_path)
    assert_that(loaded_tok.pad_token).is_equal_to("<eos>")
    assert_that(loaded_model).is_equal_to(model)
    model.to.assert_called_with("mps")


def test_load_trained_full_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full checkpoints load via AutoModelForCausalLM on CUDA."""
    tok = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    model = MagicMock()
    model.to.return_value = model
    model.config = SimpleNamespace(use_cache=False)

    monkeypatch.setattr(
        "slicktune.recipes.probe.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tok,
    )
    monkeypatch.setattr(
        "slicktune.recipes.probe.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr("slicktune.recipes.probe.resolve_dtype", lambda: torch.float32)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    loaded_model, loaded_tok = load_trained(tmp_path)
    assert_that(loaded_tok.pad_token).is_equal_to("<pad>")
    assert_that(loaded_model).is_equal_to(model)
    model.to.assert_called_with("cuda")


def test_load_trained_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU-only environments skip device moves."""
    tok = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    model = MagicMock()
    model.config = SimpleNamespace(use_cache=False)

    monkeypatch.setattr(
        "slicktune.recipes.probe.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tok,
    )
    monkeypatch.setattr(
        "slicktune.recipes.probe.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr("slicktune.recipes.probe.resolve_dtype", lambda: torch.float32)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    loaded_model, _ = load_trained(tmp_path)
    assert_that(loaded_model).is_equal_to(model)
    model.to.assert_not_called()
