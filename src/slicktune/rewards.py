"""Verifiable reward helpers for GRPO / RL objectives."""

from __future__ import annotations

import re
from typing import Any

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "yes",
        "no",
        "uses",
    }
)


def _completion_text(completion: Any) -> str:
    """Flatten a GRPO completion to plain text.

    Parameters
    ----------
    completion : Any
        Either a string or a chat message list (TRL conversational format).

    Returns
    -------
    str
        Extracted completion text.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for turn in completion:
            if isinstance(turn, dict) and "content" in turn:
                parts.append(str(turn["content"]))
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(completion)


def _keywords(needle: str) -> list[str]:
    """Extract content keywords from a ``must_contain`` needle.

    Parameters
    ----------
    needle : str
        Required substring / phrase.

    Returns
    -------
    list[str]
        Lowercased content tokens used for soft overlap rewards.
    """
    tokens = re.findall(r"[a-z0-9@./_+-]+", needle.lower())
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 2]


def substring_must_contain_reward(
    completions: list[Any],
    must_contain: list[str],
    **kwargs: Any,
) -> list[float]:
    """Score completions against required substrings.

    Full case-insensitive substring match → ``1.0``. Otherwise return the
    fraction of content keywords from ``must_contain`` that appear in the
    completion (denser signal for GRPO when the base model rarely emits the
    exact phrase).

    Parameters
    ----------
    completions : list
        Generated completions (strings or chat message lists).
    must_contain : list[str]
        Required substrings aligned with ``completions``.
    **kwargs : Any
        Unused TRL extras (``prompts``, ``trainer_state``, …).

    Returns
    -------
    list[float]
        Per-completion rewards in ``[0.0, 1.0]``.
    """
    del kwargs
    rewards: list[float] = []
    for completion, needle in zip(completions, must_contain, strict=True):
        text = _completion_text(completion)
        lowered = text.lower()
        if needle.lower() in lowered:
            rewards.append(1.0)
            continue
        keys = _keywords(needle)
        if not keys:
            rewards.append(0.0)
            continue
        hits = sum(1 for key in keys if key in lowered)
        rewards.append(hits / len(keys))
    return rewards


__all__ = ["substring_must_contain_reward"]
