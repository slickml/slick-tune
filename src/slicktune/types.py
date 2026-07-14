"""Shared base abstractions for slick-tune."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Strategy(ABC):
    """Base parameter-update strategy (how weights change).

    Parameters
    ----------
    name : str
        Strategy identifier (set by subclasses, e.g. ``\"lora\"``).
    """

    name: str = field(init=False)

    @abstractmethod
    def apply(self, model: Any) -> Any:
        """Wrap or configure ``model`` for this strategy.

        Parameters
        ----------
        model : Any
            Base causal-LM module (typically a Hugging Face model).

        Returns
        -------
        Any
            Model ready for training under this strategy.
        """

    @abstractmethod
    def load_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``from_pretrained`` that this strategy requires.

        Returns
        -------
        dict[str, Any]
            Keyword arguments passed when loading the base model.
        """


@dataclass(frozen=True)
class Objective(ABC):
    """Base training objective (what the model learns).

    Parameters
    ----------
    name : str
        Objective identifier (set by subclasses, e.g. ``\"sft\"``).
    """

    name: str = field(init=False)

    @abstractmethod
    def required_columns(self) -> list[str]:
        """Return dataset column names required by this objective.

        Returns
        -------
        list[str]
            Column names that must exist on the training dataset.
        """
