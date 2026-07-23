"""SlickML fine-tuning toolkit for LLMs."""

from importlib.metadata import PackageNotFoundError, version

from slicktune.eval import (
    HoldoutEvalResult,
    JudgeReport,
    JudgeResult,
    LLMJudge,
    SubstringJudge,
    compute_holdout_perplexity,
    run_judge_on_probes,
)
from slicktune.metrics import MetricsTracker, TrainingMetrics
from slicktune.objectives import (
    DPOObjective,
    GRPOObjective,
    KTOObjective,
    ORPOObjective,
    SFTObjective,
)
from slicktune.strategies import (
    AdaLoRAStrategy,
    DoRAStrategy,
    FullStrategy,
    LoRAStrategy,
    QLoRAStrategy,
)
from slicktune.tuner import FitResult, Tuner

try:
    __version__ = version("slicktune")
except PackageNotFoundError:  # pragma: no cover
    try:
        __version__ = version("slick-tune")
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "0.1.0"

__all__ = [
    "AdaLoRAStrategy",
    "DoRAStrategy",
    "DPOObjective",
    "FitResult",
    "FullStrategy",
    "GRPOObjective",
    "HoldoutEvalResult",
    "JudgeReport",
    "JudgeResult",
    "KTOObjective",
    "LLMJudge",
    "LoRAStrategy",
    "MetricsTracker",
    "ORPOObjective",
    "QLoRAStrategy",
    "SFTObjective",
    "SubstringJudge",
    "TrainingMetrics",
    "Tuner",
    "__version__",
    "compute_holdout_perplexity",
    "run_judge_on_probes",
]
