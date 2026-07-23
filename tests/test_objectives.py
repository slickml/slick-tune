"""Tests for objective metadata."""

from __future__ import annotations

from assertpy import assert_that

from slicktune.objectives import (
    DPOObjective,
    GRPOObjective,
    KTOObjective,
    ORPOObjective,
    SFTObjective,
)


def test_sft_required_columns() -> None:
    """SFT requires a messages column."""
    assert_that(SFTObjective().required_columns()).is_equal_to(["messages"])


def test_dpo_required_columns() -> None:
    """DPO requires preference triple columns."""
    assert_that(DPOObjective().required_columns()).is_equal_to(["prompt", "chosen", "rejected"])
    assert_that(DPOObjective(beta=0.2).beta).is_equal_to(0.2)


def test_orpo_required_columns() -> None:
    """ORPO uses the same preference columns as DPO."""
    assert_that(ORPOObjective().required_columns()).is_equal_to(["prompt", "chosen", "rejected"])


def test_kto_required_columns() -> None:
    """KTO requires unpaired preference columns."""
    assert_that(KTOObjective().required_columns()).is_equal_to(["prompt", "completion", "label"])


def test_grpo_required_columns() -> None:
    """GRPO requires prompt + verifiable substring columns."""
    assert_that(GRPOObjective().required_columns()).is_equal_to(["prompt", "must_contain"])
    assert_that(GRPOObjective(num_generations=2).num_generations).is_equal_to(2)
