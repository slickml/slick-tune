"""Tests for the Click CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from assertpy import assert_that
from click.testing import CliRunner

from slicktune.cli import cli
from slicktune.cli.main import _strategy_from_name
from slicktune.eval import HoldoutEvalResult, JudgeReport, JudgeResult
from slicktune.metrics import TrainingMetrics
from slicktune.recipes import ProbeReport, ProbeResult
from slicktune.strategies import (
    AdaLoRAStrategy,
    DoRAStrategy,
    FullStrategy,
    LoRAStrategy,
    QLoRAStrategy,
)
from slicktune.tuner import FitResult


def test_cli_package_exports() -> None:
    """Package re-exports the click group."""
    assert_that(cli).is_not_none()
    assert_that(cli.name).is_equal_to("cli")


def test_cli_version() -> None:
    """``slicktune --version`` prints the package version."""
    result = CliRunner().invoke(cli, ["--version"])
    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).contains("slicktune")
    assert_that(result.output).matches(r"\d+\.\d+\.\d+")


def test_strategy_from_name() -> None:
    """CLI strategy names map to strategy classes."""
    assert_that(_strategy_from_name("lora")).is_instance_of(LoRAStrategy)
    assert_that(_strategy_from_name("dora")).is_instance_of(DoRAStrategy)
    assert_that(_strategy_from_name("adalora")).is_instance_of(AdaLoRAStrategy)
    assert_that(_strategy_from_name("qlora")).is_instance_of(QLoRAStrategy)
    assert_that(_strategy_from_name("full")).is_instance_of(FullStrategy)


def test_strategy_from_name_unknown() -> None:
    """Unknown strategy names raise BadParameter."""
    import click

    with pytest.raises(click.BadParameter, match="Unknown strategy"):
        _strategy_from_name("not-a-strategy")


def test_train_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """train CLI wires Tuner.fit and prints metrics."""
    data = tmp_path / "data.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Yo"}]}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"

    metrics = TrainingMetrics(
        strategy="lora",
        objective="sft",
        model_id="fake",
        train_loss=0.1,
        trainable_params=10,
        total_params=100,
        eval_perplexity=2.5,
        eval_loss=0.9,
        judge_score=1.0,
    )
    fit_result = FitResult(
        output_dir=out,
        metrics=metrics,
        model=MagicMock(),
        tokenizer=MagicMock(),
    )

    fake_tuner = MagicMock()
    fake_tuner.fit.return_value = fit_result
    monkeypatch.setattr("slicktune.cli.main.Tuner", lambda **kwargs: fake_tuner)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            "--data",
            str(data),
            "--output",
            str(out),
            "--strategy",
            "dora",
            "--epochs",
            "1",
        ],
    )
    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).contains("Saved")
    assert_that(result.output).contains("perplexity=")
    fake_tuner.fit.assert_called_once()


def test_train_command_without_trainable_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train CLI prints loss only when trainable percent is missing."""
    data = tmp_path / "data.jsonl"
    data.write_text(
        '{"prompt":"a","response":"b"}\n',
        encoding="utf-8",
    )
    metrics = TrainingMetrics(
        strategy="lora",
        objective="sft",
        model_id="fake",
        train_loss=0.2,
    )
    fit_result = FitResult(
        output_dir=tmp_path / "out",
        metrics=metrics,
        model=MagicMock(),
        tokenizer=MagicMock(),
    )
    fake_tuner = MagicMock()
    fake_tuner.fit.return_value = fit_result
    monkeypatch.setattr("slicktune.cli.main.Tuner", lambda **kwargs: fake_tuner)

    result = CliRunner().invoke(
        cli,
        ["train", "--data", str(data), "--output", str(tmp_path / "out")],
    )
    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).contains("train_loss=0.2")
    assert_that(result.output).does_not_contain("trainable=")


def test_probe_command_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """probe CLI exits 0 when all probes pass."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "slicktune.cli.main.load_trained",
        lambda path: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "slicktune.cli.main.run_probes",
        lambda *args, **kwargs: ProbeReport(
            results=[
                ProbeResult(
                    prompt="Who?", must_contain="SlickML", generation="SlickML founder", passed=True
                )
            ]
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["probe", "--model-dir", str(model_dir), "--probes", str(probes)],
    )
    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).contains("100%")


def test_probe_command_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """probe CLI exits 1 when any probe fails."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "slicktune.cli.main.load_trained",
        lambda path: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "slicktune.cli.main.run_probes",
        lambda *args, **kwargs: ProbeReport(
            results=[
                ProbeResult(
                    prompt="Who?", must_contain="SlickML", generation="unknown", passed=False
                )
            ]
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["probe", "--model-dir", str(model_dir), "--probes", str(probes)],
    )
    assert_that(result.exit_code).is_equal_to(1)


def test_eval_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """eval CLI runs holdout + substring judge."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text(
        '{"prompt":"a","response":"b"}\n',
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "slicktune.cli.main.load_trained",
        lambda path: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "slicktune.cli.main.compute_holdout_perplexity",
        lambda *a, **k: HoldoutEvalResult(eval_loss=0.5, perplexity=1.65, num_examples=1),
    )
    monkeypatch.setattr(
        "slicktune.cli.main.run_judge_on_probes",
        lambda *a, **k: JudgeReport(
            results=[JudgeResult(prompt="Who?", generation="SlickML", score=1.0, rationale="ok")]
        ),
    )
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "--model-dir",
            str(model_dir),
            "--eval-data",
            str(eval_data),
            "--probes",
            str(probes),
            "--judge",
            "substring",
        ],
    )
    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).contains("Holdout")
    assert_that(result.output).contains("Judge")


def test_eval_command_llm_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """eval CLI selects LLMJudge when requested."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        '{"prompt":"Who?","must_contain":"SlickML"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "slicktune.cli.main.load_trained",
        lambda path: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "slicktune.cli.main.run_judge_on_probes",
        lambda *a, **k: JudgeReport(results=[]),
    )
    result = CliRunner().invoke(
        cli,
        ["eval", "--model-dir", str(model_dir), "--probes", str(probes), "--judge", "llm"],
    )
    assert_that(result.exit_code).is_equal_to(0)


def test_eval_command_holdout_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """eval CLI works with only --eval-data."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text('{"prompt":"a","response":"b"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "slicktune.cli.main.load_trained",
        lambda path: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(
        "slicktune.cli.main.compute_holdout_perplexity",
        lambda *a, **k: HoldoutEvalResult(eval_loss=0.5, perplexity=1.65, num_examples=1),
    )
    result = CliRunner().invoke(
        cli,
        ["eval", "--model-dir", str(model_dir), "--eval-data", str(eval_data)],
    )
    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).contains("Holdout")


def test_eval_command_requires_inputs(tmp_path: Path) -> None:
    """eval without data/probes errors."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result = CliRunner().invoke(cli, ["eval", "--model-dir", str(model_dir)])
    assert_that(result.exit_code).is_not_equal_to(0)
