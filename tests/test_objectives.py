"""Tests for objective metadata."""

from __future__ import annotations

from assertpy import assert_that

from slicktune.objectives import DPOObjective, SFTObjective


def test_sft_required_columns() -> None:
    """SFT requires a messages column."""
    assert_that(SFTObjective().required_columns()).is_equal_to(["messages"])


def test_dpo_required_columns() -> None:
    """DPO placeholder documents preference columns."""
    assert_that(DPOObjective().required_columns()).contains("chosen", "rejected")
