"""Dataset loaders for fine-tuning objectives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about Amirhessam Tahmassebi, SlickML, and slick-tune "
    "accurately and concisely."
)


def _with_system_message(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Ensure a chat example starts with our system prompt.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Chat turns.
    system_prompt : str, optional
        System content used when the example has no system turn.

    Returns
    -------
    list[dict[str, Any]]
        Messages with a leading system turn (replacing any existing one).
    """
    rest = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}, *rest]


def _normalize_example(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSONL row into a chat ``messages`` example.

    Parameters
    ----------
    raw : dict[str, Any]
        Raw row with ``messages``, or ``prompt``/``response``, or
        ``instruction``/``output``.

    Returns
    -------
    dict[str, Any]
        Mapping with a single ``messages`` list of role/content dicts.

    Raises
    ------
    ValueError
        If the row cannot be mapped to a chat example.
    """
    if "messages" in raw:
        messages = raw["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        return {"messages": _with_system_message(messages)}

    if "prompt" in raw and "response" in raw:
        return {
            "messages": _with_system_message(
                [
                    {"role": "user", "content": str(raw["prompt"])},
                    {"role": "assistant", "content": str(raw["response"])},
                ]
            )
        }

    if "instruction" in raw and "output" in raw:
        user = str(raw["instruction"])
        if raw.get("input"):
            user = f"{user}\n\n{raw['input']}"
        return {
            "messages": _with_system_message(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": str(raw["output"])},
                ]
            )
        }

    raise ValueError("Each example needs messages, or prompt+response, or instruction+output")


def load_sft_jsonl(path: str | Path) -> Dataset:
    """Load an SFT JSONL file into a Hugging Face ``Dataset``.

    Parameters
    ----------
    path : str or Path
        Path to a ``.jsonl`` file. Each line is one training example.

    Returns
    -------
    datasets.Dataset
        Dataset with a ``messages`` column.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty or a row is invalid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"SFT data not found: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            rows.append(_normalize_example(raw))

    if not rows:
        raise ValueError(f"No examples found in {file_path}")

    return Dataset.from_list(rows)


def load_probe_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Load probe questions used to verify fine-tuning worked.

    Parameters
    ----------
    path : str or Path
        JSONL with ``prompt`` and ``must_contain`` (substring checks).

    Returns
    -------
    list[dict[str, str]]
        Probe rows with ``prompt`` and ``must_contain``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a row is missing required keys.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Probe data not found: {file_path}")

    probes: list[dict[str, str]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if "prompt" not in raw or "must_contain" not in raw:
                raise ValueError(f"Probe line {line_no} needs prompt and must_contain")
            probes.append(
                {
                    "prompt": str(raw["prompt"]),
                    "must_contain": str(raw["must_contain"]),
                }
            )
    return probes


def _as_text(value: Any, *, field_name: str, line_no: int) -> str:
    """Coerce a preference field to a plain string.

    Parameters
    ----------
    value : Any
        Raw field value (string or chat message list).
    field_name : str
        Column name for error messages.
    line_no : int
        JSONL line number for error messages.

    Returns
    -------
    str
        Flattened text.

    Raises
    ------
    ValueError
        If ``value`` cannot be converted.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Line {line_no}: {field_name} must be non-empty")
        return text
    if isinstance(value, list):
        parts: list[str] = []
        for turn in value:
            if not isinstance(turn, dict) or "content" not in turn:
                raise ValueError(f"Line {line_no}: {field_name} list items need content")
            parts.append(str(turn["content"]))
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError(f"Line {line_no}: {field_name} must be non-empty")
        return text
    raise ValueError(f"Line {line_no}: {field_name} must be a string or message list")


def load_preference_jsonl(path: str | Path) -> Dataset:
    """Load a DPO/ORPO preference JSONL file.

    Parameters
    ----------
    path : str or Path
        JSONL with ``prompt``, ``chosen``, and ``rejected`` per line.

    Returns
    -------
    datasets.Dataset
        Dataset with string preference columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty or a row is invalid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Preference data not found: {file_path}")

    rows: list[dict[str, str]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            missing = [k for k in ("prompt", "chosen", "rejected") if k not in raw]
            if missing:
                raise ValueError(f"Line {line_no} missing required keys: {', '.join(missing)}")
            rows.append(
                {
                    "prompt": _as_text(raw["prompt"], field_name="prompt", line_no=line_no),
                    "chosen": _as_text(raw["chosen"], field_name="chosen", line_no=line_no),
                    "rejected": _as_text(raw["rejected"], field_name="rejected", line_no=line_no),
                }
            )

    if not rows:
        raise ValueError(f"No examples found in {file_path}")

    return Dataset.from_list(rows)


def load_kto_jsonl(path: str | Path) -> Dataset:
    """Load a KTO unpaired-preference JSONL file.

    Parameters
    ----------
    path : str or Path
        JSONL with ``prompt``, ``completion``, and boolean ``label`` per line.

    Returns
    -------
    datasets.Dataset
        Dataset with KTO columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty or a row is invalid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"KTO data not found: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            missing = [k for k in ("prompt", "completion", "label") if k not in raw]
            if missing:
                raise ValueError(f"Line {line_no} missing required keys: {', '.join(missing)}")
            label = raw["label"]
            if not isinstance(label, bool):
                raise ValueError(f"Line {line_no}: label must be a boolean")
            rows.append(
                {
                    "prompt": _as_text(raw["prompt"], field_name="prompt", line_no=line_no),
                    "completion": _as_text(
                        raw["completion"], field_name="completion", line_no=line_no
                    ),
                    "label": label,
                }
            )

    if not rows:
        raise ValueError(f"No examples found in {file_path}")

    return Dataset.from_list(rows)


def load_grpo_jsonl(path: str | Path) -> Dataset:
    """Load a GRPO / verifiable-RL JSONL file.

    Parameters
    ----------
    path : str or Path
        JSONL with ``prompt`` and ``must_contain`` (or ``solution`` alias) per line.

    Returns
    -------
    datasets.Dataset
        Dataset with ``prompt`` and ``must_contain`` columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty or a row is invalid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"GRPO data not found: {file_path}")

    rows: list[dict[str, str]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            if "prompt" not in raw:
                raise ValueError(f"Line {line_no} missing required key: prompt")
            needle = raw.get("must_contain", raw.get("solution"))
            if needle is None:
                raise ValueError(f"Line {line_no} missing required key: must_contain (or solution)")
            rows.append(
                {
                    "prompt": _as_text(raw["prompt"], field_name="prompt", line_no=line_no),
                    "must_contain": _as_text(needle, field_name="must_contain", line_no=line_no),
                }
            )

    if not rows:
        raise ValueError(f"No examples found in {file_path}")

    return Dataset.from_list(rows)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "load_grpo_jsonl",
    "load_kto_jsonl",
    "load_preference_jsonl",
    "load_probe_jsonl",
    "load_sft_jsonl",
]
