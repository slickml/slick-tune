"""Tests for SFT JSONL loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
from assertpy import assert_that

from slicktune.data import (
    load_kto_jsonl,
    load_preference_jsonl,
    load_probe_jsonl,
    load_sft_jsonl,
)


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
    prefs = load_preference_jsonl(data_dir / "about_amir.prefs.jsonl")
    kto = load_kto_jsonl(data_dir / "about_amir.kto.jsonl")
    assert_that(len(ds)).is_greater_than(5)
    assert_that(len(holdout)).is_greater_than(3)
    assert_that(probes).is_not_empty()
    assert_that(len(prefs)).is_greater_than(3)
    assert_that(len(kto)).is_greater_than(3)
    train_prompts = {
        turn["content"] for row in ds for turn in row["messages"] if turn.get("role") == "user"
    }
    eval_prompts = {
        turn["content"] for row in holdout for turn in row["messages"] if turn.get("role") == "user"
    }
    assert_that(train_prompts.intersection(eval_prompts)).is_empty()


def test_load_preference_jsonl(tmp_path: Path) -> None:
    """Load DPO/ORPO preference triples."""
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"prompt":"Who?","chosen":"SlickML founder","rejected":"Wrong person"}\n',
        encoding="utf-8",
    )
    ds = load_preference_jsonl(path)
    assert_that(len(ds)).is_equal_to(1)
    assert_that(ds[0]["chosen"]).is_equal_to("SlickML founder")


def test_load_preference_jsonl_missing_keys(tmp_path: Path) -> None:
    """Reject preference rows without chosen/rejected."""
    path = tmp_path / "prefs.jsonl"
    path.write_text('{"prompt":"Who?","chosen":"ok"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="rejected"):
        load_preference_jsonl(path)


def test_load_preference_jsonl_missing_file() -> None:
    """Raise when preference path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_preference_jsonl("/tmp/does-not-exist-prefs.jsonl")


def test_load_kto_jsonl(tmp_path: Path) -> None:
    """Load KTO unpaired preference rows."""
    path = tmp_path / "kto.jsonl"
    path.write_text(
        '{"prompt":"Who?","completion":"Founder","label":true}\n'
        '{"prompt":"Who?","completion":"Wrong","label":false}\n',
        encoding="utf-8",
    )
    ds = load_kto_jsonl(path)
    assert_that(len(ds)).is_equal_to(2)
    assert_that(ds[0]["label"]).is_true()
    assert_that(ds[1]["label"]).is_false()


def test_load_kto_jsonl_non_bool_label(tmp_path: Path) -> None:
    """Reject non-boolean KTO labels."""
    path = tmp_path / "kto.jsonl"
    path.write_text(
        '{"prompt":"Who?","completion":"Founder","label":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="boolean"):
        load_kto_jsonl(path)


def test_load_preference_jsonl_message_list(tmp_path: Path) -> None:
    """Preference fields may be chat message lists."""
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        '{"prompt":[{"role":"user","content":"Who?"}],'
        '"chosen":[{"role":"assistant","content":"Good"}],'
        '"rejected":[{"role":"assistant","content":"Bad"}]}\n',
        encoding="utf-8",
    )
    ds = load_preference_jsonl(path)
    assert_that(ds[0]["prompt"]).is_equal_to("Who?")
    assert_that(ds[0]["chosen"]).is_equal_to("Good")


def test_preference_loader_error_paths(tmp_path: Path) -> None:
    """Cover preference/KTO validation failures."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No examples"):
        load_preference_jsonl(empty)
    with pytest.raises(ValueError, match="No examples"):
        load_kto_jsonl(empty)

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_preference_jsonl(bad_json)
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_kto_jsonl(bad_json)

    non_obj = tmp_path / "arr.jsonl"
    non_obj.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_preference_jsonl(non_obj)
    with pytest.raises(ValueError, match="JSON object"):
        load_kto_jsonl(non_obj)

    blank_field = tmp_path / "blank.jsonl"
    blank_field.write_text(
        '{"prompt":" ","chosen":"ok","rejected":"bad"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty"):
        load_preference_jsonl(blank_field)

    bad_list = tmp_path / "list.jsonl"
    bad_list.write_text(
        '{"prompt":[1],"chosen":"ok","rejected":"bad"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content"):
        load_preference_jsonl(bad_list)

    empty_list = tmp_path / "elist.jsonl"
    empty_list.write_text(
        '{"prompt":[{"role":"user","content":"  "}],"chosen":"ok","rejected":"bad"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty"):
        load_preference_jsonl(empty_list)

    wrong_type = tmp_path / "num.jsonl"
    wrong_type.write_text(
        '{"prompt":1,"chosen":"ok","rejected":"bad"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="string or message list"):
        load_preference_jsonl(wrong_type)

    with pytest.raises(FileNotFoundError):
        load_kto_jsonl("/tmp/does-not-exist-kto.jsonl")

    kto_missing = tmp_path / "kto_miss.jsonl"
    kto_missing.write_text('{"prompt":"Who?","completion":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        load_kto_jsonl(kto_missing)
