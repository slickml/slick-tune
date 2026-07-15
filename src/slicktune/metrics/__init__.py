"""Training and evaluation metrics tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(kw_only=True)
class TrainingMetrics:
    """Snapshot of metrics for one training run.

    Parameters
    ----------
    strategy : str
        Parameter strategy name (e.g. ``"lora"``).
    objective : str
        Objective name (e.g. ``"sft"``).
    model_id : str
        Base model id or path.
    train_loss : float | None, optional
        Final reported training loss, by default None.
    eval_loss : float | None, optional
        Final evaluation loss if computed, by default None.
    train_runtime_sec : float | None, optional
        Wall-clock training time in seconds, by default None.
    train_samples_per_second : float | None, optional
        Throughput reported by the trainer, by default None.
    trainable_params : int | None, optional
        Number of trainable parameters, by default None.
    total_params : int | None, optional
        Total parameter count, by default None.
    probe_pass_rate : float | None, optional
        Fraction of probe checks that passed after training, by default None.
    eval_perplexity : float | None, optional
        Holdout perplexity from Phase-2 eval, by default None.
    judge_score : float | None, optional
        Mean judge score in ``[0, 1]``, by default None.
    extras : dict[str, Any], optional
        Additional key/value metrics, by default empty.
    """

    strategy: str
    objective: str
    model_id: str
    train_loss: float | None = None
    eval_loss: float | None = None
    train_runtime_sec: float | None = None
    train_samples_per_second: float | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    probe_pass_rate: float | None = None
    eval_perplexity: float | None = None
    judge_score: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def trainable_percent(self) -> float | None:
        """Return trainable parameters as a percent of total.

        Returns
        -------
        float or None
            Percent trainable, or None if counts are missing.
        """
        if self.trainable_params is None or self.total_params in (None, 0):
            return None
        return 100.0 * self.trainable_params / self.total_params


@dataclass(kw_only=True)
class MetricsTracker:
    """Collect and persist metrics across a run.

    Parameters
    ----------
    output_dir : str or Path
        Directory where ``metrics.json`` is written.
    """

    output_dir: Path

    def __post_init__(self) -> None:
        """Ensure ``output_dir`` is a ``Path`` and exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, metrics: TrainingMetrics) -> Path:
        """Write metrics to ``metrics.json``.

        Parameters
        ----------
        metrics : TrainingMetrics
            Metrics snapshot to persist.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        path = self.output_dir / "metrics.json"
        payload = asdict(metrics)
        payload["trainable_percent"] = metrics.trainable_percent
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load(self) -> TrainingMetrics:
        """Load metrics previously written by :meth:`save`.

        Returns
        -------
        TrainingMetrics
            Restored metrics object.

        Raises
        ------
        FileNotFoundError
            If ``metrics.json`` is missing.
        """
        path = self.output_dir / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"No metrics at {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("trainable_percent", None)
        return TrainingMetrics(**raw)


def count_parameters(model: Any) -> tuple[int, int]:
    """Count trainable and total parameters.

    Parameters
    ----------
    model : Any
        PyTorch module.

    Returns
    -------
    tuple[int, int]
        ``(trainable_params, total_params)``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


__all__ = ["MetricsTracker", "TrainingMetrics", "count_parameters"]
