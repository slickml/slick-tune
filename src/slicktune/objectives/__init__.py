"""Training objectives (what the model learns)."""

from __future__ import annotations

from dataclasses import dataclass, field

from slicktune.types import Objective


@dataclass(frozen=True, kw_only=True)
class SFTObjective(Objective):
    """Supervised fine-tuning on instruction / chat pairs."""

    name: str = field(default="sft", init=False)

    def required_columns(self) -> list[str]:
        """Return required dataset columns for SFT.

        Returns
        -------
        list[str]
            Chat ``messages`` column name.
        """
        return ["messages"]


# Phase-3 placeholder so the public surface documents the roadmap.
@dataclass(frozen=True, kw_only=True)
class DPOObjective(Objective):
    """Direct Preference Optimization (Phase 3).

    Parameters
    ----------
    beta : float, optional
        KL penalty coefficient, by default 0.1.
    """

    name: str = field(default="dpo", init=False)
    beta: float = 0.1

    def required_columns(self) -> list[str]:
        """Return required preference columns.

        Returns
        -------
        list[str]
            Preference triple column names.
        """
        return ["prompt", "chosen", "rejected"]


__all__ = ["DPOObjective", "SFTObjective"]
