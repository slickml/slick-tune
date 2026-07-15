"""Tests for Phase-2 eval harness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from assertpy import assert_that
from datasets import Dataset

from slicktune.eval import (
    JudgeReport,
    JudgeResult,
    LLMJudge,
    SubstringJudge,
    _parse_score_0_to_10,
    compute_holdout_perplexity,
    run_judge_on_probes,
)


def test_substring_judge_pass_fail() -> None:
    """Substring judge scores exact containment."""
    judge = SubstringJudge()
    ok = judge.judge(prompt="Who?", generation="SlickML founder", must_contain="SlickML")
    bad = judge.judge(prompt="Who?", generation="unknown", must_contain="SlickML")
    assert_that(ok.score).is_equal_to(1.0)
    assert_that(bad.score).is_equal_to(0.0)


def test_judge_report_mean() -> None:
    """Mean score averages judgments."""
    report = JudgeReport(
        results=[
            JudgeResult(prompt="a", generation="x", score=1.0, rationale="ok"),
            JudgeResult(prompt="b", generation="y", score=0.0, rationale="no"),
        ]
    )
    assert_that(report.mean_score).is_equal_to(0.5)
    assert_that(JudgeReport().mean_score).is_equal_to(0.0)


def test_parse_score_0_to_10() -> None:
    """Parser finds a real 0–10 score and ignores scale echoes."""
    assert_that(_parse_score_0_to_10("score is 8")).is_equal_to(0.8)
    assert_that(_parse_score_0_to_10("10/10")).is_equal_to(1.0)
    assert_that(_parse_score_0_to_10("no number")).is_equal_to(0.0)
    # Echoing the scale must not become score 0 via the leading digit.
    assert_that(_parse_score_0_to_10("0-10")).is_equal_to(0.0)
    assert_that(_parse_score_0_to_10("0-10: SlickML is great")).is_equal_to(0.0)
    assert_that(_parse_score_0_to_10("0 to 10\n8")).is_equal_to(0.8)
    assert_that(_parse_score_0_to_10("SCORE: 9")).is_equal_to(0.9)
    assert_that(_parse_score_0_to_10("7")).is_equal_to(0.7)


def test_llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM judge parses a numeric score from constrained score generation."""
    monkeypatch.setattr(
        "slicktune.eval._generate_score_0_to_10",
        lambda **kwargs: "7",
    )
    result = LLMJudge(model=MagicMock(), tokenizer=MagicMock()).judge(prompt="q", generation="a")
    assert_that(result.score).is_equal_to(0.7)


def test_single_token_ids_for_text() -> None:
    """Helper keeps only exact single-token encodings."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, add_special_tokens=False: [42] if text == "8" else [1, 2]
    from slicktune.eval import _single_token_ids_for_text

    assert_that(_single_token_ids_for_text(tok, "8")).is_equal_to([42])
    assert_that(_single_token_ids_for_text(tok, "10")).is_equal_to([])


def test_generate_score_falls_back_without_digit_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If digits are not single tokens, fall back to generate_reply."""
    from slicktune.eval import _generate_score_0_to_10

    tok = MagicMock()
    tok.chat_template = None
    tok.encode.return_value = [9, 9]
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([param])
    model.config = SimpleNamespace(use_cache=False)

    monkeypatch.setattr(
        "slicktune.eval.generate_reply",
        lambda **kwargs: "9",
    )
    monkeypatch.setattr("slicktune.eval._single_token_ids_for_text", lambda *a, **k: [])
    out = _generate_score_0_to_10(model=model, tokenizer=tok, rubric="SCORE:")
    assert_that(out).is_equal_to("9")


def test_generate_score_constrained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constrained path decodes the generated score tokens."""
    from slicktune.eval import _generate_score_0_to_10

    tok = MagicMock()
    tok.pad_token_id = None
    tok.eos_token_id = 99
    tok.return_value = {"input_ids": torch.tensor([[5, 6]])}
    tok.decode.return_value = "8"

    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([param])
    model.config = SimpleNamespace(use_cache=False)
    model.generate.return_value = torch.tensor([[5, 6, 8]])

    monkeypatch.setattr(
        "slicktune.eval._single_token_ids_for_text",
        lambda _tokenizer, text: [int(text.strip())] if text.strip().isdigit() else [],
    )
    out = _generate_score_0_to_10(model=model, tokenizer=tok, rubric="RATE\nSCORE:")
    assert_that(out).is_equal_to("8")
    # Tokenizer should see the SCORE: completion prefix.
    assert_that(tok.call_args.args[0]).contains("SCORE:")
    kwargs = model.generate.call_args.kwargs
    allow = kwargs["prefix_allowed_tokens_fn"]
    first = allow(0, torch.tensor([5, 6]))
    assert_that(first).contains(8)
    second = allow(0, torch.tensor([5, 6, 1]))
    assert_that(second).contains(0)
    assert_that(second).contains(99)
    done = allow(0, torch.tensor([5, 6, 8]))
    assert_that(done).is_equal_to([99])


def test_generate_score_appends_score_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rubrics without SCORE: get the prefix appended."""
    from slicktune.eval import _generate_score_0_to_10

    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 99
    tok.return_value = {"input_ids": torch.tensor([[1]])}
    tok.decode.return_value = "5"
    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([param])
    model.config = SimpleNamespace(use_cache=False)
    model.generate.return_value = torch.tensor([[1, 5]])
    monkeypatch.setattr(
        "slicktune.eval._single_token_ids_for_text",
        lambda _tokenizer, text: [int(text.strip())] if text.strip().isdigit() else [],
    )
    _generate_score_0_to_10(model=model, tokenizer=tok, rubric="rate this")
    assert_that(tok.call_args.args[0]).ends_with("SCORE:")


def test_generate_score_allow_without_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow-fn falls back when eos_token_id is missing."""
    from slicktune.eval import _generate_score_0_to_10

    tok = MagicMock()
    tok.chat_template = None
    tok.pad_token_id = 0
    tok.eos_token_id = None
    tok.return_value = {"input_ids": torch.tensor([[5, 6]])}
    tok.decode.return_value = "10"

    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([param])
    model.config = SimpleNamespace(use_cache=False)
    model.generate.return_value = torch.tensor([[5, 6, 1, 0]])

    monkeypatch.setattr(
        "slicktune.eval._single_token_ids_for_text",
        lambda _tokenizer, text: [int(text.strip())] if text.strip().isdigit() else [],
    )
    out = _generate_score_0_to_10(model=model, tokenizer=tok, rubric="SCORE:")
    assert_that(out).is_equal_to("10")
    allow = model.generate.call_args.kwargs["prefix_allowed_tokens_fn"]
    after_one = allow(0, torch.tensor([5, 6, 1]))
    assert_that(after_one).is_equal_to([0])
    after_eight = allow(0, torch.tensor([5, 6, 8]))
    assert_that(after_eight).contains(8)


def test_compute_holdout_perplexity() -> None:
    """Holdout perplexity averages finite losses."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "hello world"
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    loss = MagicMock()
    loss.detach.return_value.cpu.return_value = torch.tensor(0.0)
    # float(tensor) works for 0-dim; use real tensor via side effect
    outputs = SimpleNamespace(loss=torch.tensor(0.693147))

    param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([param])
    model.return_value = outputs
    model.config = SimpleNamespace(use_cache=False)

    dataset = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]}]
    )
    result = compute_holdout_perplexity(model=model, tokenizer=tokenizer, data=dataset)
    assert_that(result.num_examples).is_equal_to(1)
    assert_that(result.eval_loss).is_greater_than(0.0)
    assert_that(result.perplexity).is_greater_than(1.0)


def test_compute_holdout_empty_raises() -> None:
    """Empty holdout dataset raises."""
    with pytest.raises(ValueError, match="empty"):
        compute_holdout_perplexity(
            model=MagicMock(),
            tokenizer=MagicMock(),
            data=Dataset.from_list([]),
        )


def test_compute_holdout_skips_non_finite_loss() -> None:
    """Non-finite losses are skipped until none remain."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "hello world"
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    outputs = SimpleNamespace(loss=torch.tensor(float("nan")))
    param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([param])
    model.return_value = outputs
    model.config = SimpleNamespace(use_cache=False)
    dataset = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]}]
    )
    with pytest.raises(ValueError, match="finite"):
        compute_holdout_perplexity(model=model, tokenizer=tokenizer, data=dataset)


def test_compute_holdout_skips_short_sequences() -> None:
    """Sequences shorter than 2 tokens are skipped."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "x"
    tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
    param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([param])
    model.config = SimpleNamespace(use_cache=False)
    dataset = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]}]
    )
    with pytest.raises(ValueError, match="finite"):
        compute_holdout_perplexity(model=model, tokenizer=tokenizer, data=dataset)


def test_run_judge_on_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe loop feeds generations into the judge."""
    path = tmp_path / "probes.jsonl"
    path.write_text('{"prompt":"Who?","must_contain":"SlickML"}\n', encoding="utf-8")
    monkeypatch.setattr("slicktune.eval.generate_reply", lambda *a, **k: "SlickML")
    model = MagicMock()
    model.config = SimpleNamespace(use_cache=False)
    report = run_judge_on_probes(
        model=model, tokenizer=MagicMock(), probe_path=path, judge=SubstringJudge()
    )
    assert_that(report.mean_score).is_equal_to(1.0)
