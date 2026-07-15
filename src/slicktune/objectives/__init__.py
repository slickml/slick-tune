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


@dataclass(frozen=True, kw_only=True)
class DPOObjective(Objective):
    """Direct Preference Optimization (TRL :class:`~trl.DPOTrainer`).

    Parameters
    ----------
    beta : float, optional
        KL penalty coefficient, by default 0.1.
    loss_type : str, optional
        TRL DPO loss type, by default ``\"sigmoid\"``.
    """

    name: str = field(default="dpo", init=False)
    beta: float = 0.1
    loss_type: str = "sigmoid"

    def required_columns(self) -> list[str]:
        """Return required preference columns.

        Returns
        -------
        list[str]
            Preference triple column names.
        """
        return ["prompt", "chosen", "rejected"]


@dataclass(frozen=True, kw_only=True)
class ORPOObjective(Objective):
    """Odds Ratio Preference Optimization (TRL experimental ORPO).

    Parameters
    ----------
    beta : float, optional
        Odds-ratio penalty coefficient, by default 0.1.
    """

    name: str = field(default="orpo", init=False)
    beta: float = 0.1

    def required_columns(self) -> list[str]:
        """Return required preference columns.

        Returns
        -------
        list[str]
            Preference triple column names (same shape as DPO).
        """
        return ["prompt", "chosen", "rejected"]


@dataclass(frozen=True, kw_only=True)
class KTOObjective(Objective):
    """Kahneman–Tversky Optimization (TRL :class:`~trl.KTOTrainer`).

    Parameters
    ----------
    beta : float, optional
        KL penalty coefficient, by default 0.1.
    desirable_weight : float, optional
        Weight for desirable (``label=True``) examples, by default 1.0.
    undesirable_weight : float, optional
        Weight for undesirable (``label=False``) examples, by default 1.0.
    """

    name: str = field(default="kto", init=False)
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    def required_columns(self) -> list[str]:
        """Return required KTO columns.

        Returns
        -------
        list[str]
            Unpaired preference column names.
        """
        return ["prompt", "completion", "label"]


__all__ = ["DPOObjective", "KTOObjective", "ORPOObjective", "SFTObjective"]
