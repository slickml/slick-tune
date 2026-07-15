"""Tests for SFT JSONL loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
from assertpy import assert_that

from slicktune.data import load_probe_jsonl, load_sft_jsonl


def test_load_sft_jsonl_messages_format(tmp_path: Path) -> None:
    """Load chat-style messages rows and inject a system turn."""
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}\n',
        encoding="utf-8",
    )
    ds = load_sft_jsonl(path)
    assert_that(len(ds)).is_equal_to(1)
    assert_that(ds[0]["messages"][0]["role"]).is_equal_to("system")
    assert_that(ds[0]["messages"][1]["content"]).is_equal_to("Hi")


def test_load_sft_jsonl_prompt_response(tmp_path: Path) -> None:
    """Normalize prompt/response rows into messages."""
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"prompt":"Who?","response":"Amirhessam"}\n',
        encoding="utf-8",
    )
    ds = load_sft_jsonl(path)
    assert_that(ds[0]["messages"][2]["content"]).is_equal_to("Amirhessam")


def test_load_sft_jsonl_instruction_output(tmp_path: Path) -> None:
    """Normalize instruction/output rows into messages."""
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"instruction":"Say name","input":"founder","output":"Amirhessam"}\n',
        encoding="utf-8",
    )
    ds = load_sft_jsonl(path)
    assert_that(ds[0]["messages"][1]["content"]).contains("Say name")
    assert_that(ds[0]["messages"][1]["content"]).contains("founder")


def test_load_sft_jsonl_instruction_without_input(tmp_path: Path) -> None:
    """Instruction/output works when input is omitted."""
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"instruction":"Say name","output":"Amirhessam"}\n',
        encoding="utf-8",
    )
    ds = load_sft_jsonl(path)
    assert_that(ds[0]["messages"][1]["content"]).is_equal_to("Say name")


def test_load_sft_jsonl_empty_messages(tmp_path: Path) -> None:
    """Reject empty messages lists."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_sft_jsonl(path)


def test_load_sft_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines are ignored between valid rows."""
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '\n{"prompt":"Who?","response":"Amirhessam"}\n\n',
        encoding="utf-8",
    )
    ds = load_sft_jsonl(path)
    assert_that(len(ds)).is_equal_to(1)


def test_load_sft_jsonl_invalid_json(tmp_path: Path) -> None:
    """Reject malformed JSON lines."""
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_sft_jsonl(path)


def test_load_sft_jsonl_non_object_row(tmp_path: Path) -> None:
    """Reject JSON arrays / scalars as rows."""
    path = tmp_path / "bad.jsonl"
    path.write_text("[1,2,3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_sft_jsonl(path)


def test_load_sft_jsonl_empty_file(tmp_path: Path) -> None:
    """Reject files with no examples."""
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No examples"):
        load_sft_jsonl(path)


def test_load_sft_jsonl_missing_file() -> None:
    """Raise when the path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_sft_jsonl("/tmp/does-not-exist-slick-tune.jsonl")


def test_load_sft_jsonl_invalid_row(tmp_path: Path) -> None:
    """Raise when a row has no supported schema."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"foo":1}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_sft_jsonl(path)


def test_load_probe_jsonl(tmp_path: Path) -> None:
    """Load probe rows with must_contain checks."""
    path = tmp_path / "probes.jsonl"
    path.write_text(
        '\n{"prompt":"Who?","must_contain":"SlickML"}\n\n',
        encoding="utf-8",
    )
    probes = load_probe_jsonl(path)
    assert_that(probes).is_length(1)
    assert_that(probes[0]["must_contain"]).is_equal_to("SlickML")


def test_load_probe_jsonl_missing_file() -> None:
    """Raise when probe path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_probe_jsonl("/tmp/does-not-exist-probes.jsonl")


def test_load_probe_jsonl_missing_keys(tmp_path: Path) -> None:
    """Raise when probe rows omit required keys."""
    path = tmp_path / "probes.jsonl"
    path.write_text('{"prompt":"Who?"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must_contain"):
        load_probe_jsonl(path)


def test_example_about_amir_dataset_loads() -> None:
    """Ship example personal dataset must parse cleanly."""
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "examples" / "data"
    ds = load_sft_jsonl(data_dir / "about_amir.jsonl")
    holdout = load_sft_jsonl(data_dir / "about_amir.eval.jsonl")
    probes = load_probe_jsonl(data_dir / "about_amir.probes.jsonl")
    assert_that(len(ds)).is_greater_than(5)
    assert_that(len(holdout)).is_greater_than(3)
    assert_that(probes).is_not_empty()
    train_prompts = {
        turn["content"] for row in ds for turn in row["messages"] if turn.get("role") == "user"
    }
    eval_prompts = {
        turn["content"] for row in holdout for turn in row["messages"] if turn.get("role") == "user"
    }
    assert_that(train_prompts.intersection(eval_prompts)).is_empty()
