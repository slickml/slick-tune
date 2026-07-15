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


__all__ = ["DEFAULT_SYSTEM_PROMPT", "load_probe_jsonl", "load_sft_jsonl"]
