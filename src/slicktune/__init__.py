"""SlickML fine-tuning toolkit for LLMs."""

from importlib.metadata import PackageNotFoundError, version

from slicktune.metrics import MetricsTracker, TrainingMetrics
from slicktune.objectives import SFTObjective
from slicktune.strategies import FullStrategy, LoRAStrategy, QLoRAStrategy
from slicktune.tuner import FitResult, Tuner

try:
    __version__ = version("slick-tune")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "FitResult",
    "FullStrategy",
    "LoRAStrategy",
    "MetricsTracker",
    "QLoRAStrategy",
    "SFTObjective",
    "TrainingMetrics",
    "Tuner",
    "__version__",
]
