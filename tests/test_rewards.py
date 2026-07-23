"""Tests for verifiable GRPO rewards."""

from __future__ import annotations

from assertpy import assert_that

from slicktune.rewards import substring_must_contain_reward


def test_substring_must_contain_reward_string_completions() -> None:
    """Plain-string completions are scored case-insensitively."""
    rewards = substring_must_contain_reward(
        completions=["Amirhessam is the founder of SlickML.", "No idea."],
        must_contain=["founder of SlickML", "founder of SlickML"],
    )
    assert_that(rewards).is_equal_to([1.0, 0.0])


def test_substring_must_contain_reward_chat_completions() -> None:
    """Conversational completions extract the assistant content."""
    rewards = substring_must_contain_reward(
        completions=[[{"role": "assistant", "content": "admin@slickml.com"}]],
        must_contain=["admin@slickml.com"],
    )
    assert_that(rewards).is_equal_to([1.0])


def test_substring_must_contain_reward_fallback_shapes() -> None:
    """Non-dict list items and scalar completions coerce to text."""
    rewards = substring_must_contain_reward(
        completions=[["plain-turn", 42], 7],
        must_contain=["plain-turn", "7"],
    )
    assert_that(rewards).is_equal_to([1.0, 1.0])


def test_substring_must_contain_reward_partial_keywords() -> None:
    """Partial keyword overlap gives a dense reward in (0, 1)."""
    rewards = substring_must_contain_reward(
        completions=["SlickML is a machine learning library by Google."],
        must_contain=["open-source machine learning"],
    )
    assert_that(len(rewards)).is_equal_to(1)
    assert_that(rewards[0]).is_greater_than(0.0)
    assert_that(rewards[0]).is_less_than(1.0)


def test_substring_must_contain_reward_stopword_only_needle() -> None:
    """Needles with only stopwords / short tokens score 0 when unmatched."""
    rewards = substring_must_contain_reward(
        completions=["hello world"],
        must_contain=["a of the"],
    )
    assert_that(rewards).is_equal_to([0.0])
