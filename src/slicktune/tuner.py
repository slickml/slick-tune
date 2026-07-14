"""High-level Tuner API composing model × strategy × objective × data × metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from slicktune.data import load_sft_jsonl
from slicktune.metrics import MetricsTracker, TrainingMetrics, count_parameters
from slicktune.models import load_model, load_tokenizer
from slicktune.objectives import SFTObjective
from slicktune.types import Objective, Strategy


@dataclass
class FitResult:
    """Result of a completed :meth:`Tuner.fit` call.

    Parameters
    ----------
    output_dir : Path
        Directory containing the saved adapter / model and metrics.
    metrics : TrainingMetrics
        Collected training metrics.
    model : Any
        Trained model (PEFT or full).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used during training.
    """

    output_dir: Path
    metrics: TrainingMetrics
    model: Any
    tokenizer: PreTrainedTokenizerBase


@dataclass
class Tuner:
    """Composable fine-tuning entry point.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or local path.
    strategy : Strategy
        Parameter-update strategy (LoRA, QLoRA, full, ...).
    objective : Objective
        Training objective (currently SFT for Phase 1).
    output_dir : str or Path
        Where checkpoints, adapter weights, and metrics are written.
    max_seq_length : int, optional
        Maximum sequence length, by default 512.
    num_train_epochs : float, optional
        Number of epochs, by default 3.0.
    per_device_train_batch_size : int, optional
        Batch size per device, by default 1.
    gradient_accumulation_steps : int, optional
        Gradient accumulation steps, by default 4.
    learning_rate : float, optional
        Learning rate, by default 2e-4.
    logging_steps : int, optional
        Logging frequency, by default 1.
    save_steps : int, optional
        Checkpoint frequency, by default 50.
    seed : int, optional
        Random seed, by default 42.
    """

    model_id: str
    strategy: Strategy
    objective: Objective
    output_dir: str | Path
    max_seq_length: int = 512
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    logging_steps: int = 1
    save_steps: int = 50
    seed: int = 42

    def fit(self, data: str | Path | Dataset) -> FitResult:
        """Run fine-tuning.

        Parameters
        ----------
        data : str or Path or Dataset
            Path to SFT JSONL, or an in-memory dataset with ``messages``.

        Returns
        -------
        FitResult
            Trained artifacts and metrics.

        Raises
        ------
        TypeError
            If the objective is not yet supported for training.
        ValueError
            If required dataset columns are missing.
        """
        if not isinstance(self.objective, SFTObjective):
            raise TypeError(
                f"Objective '{self.objective.name}' is not implemented yet. "
                "Phase 1 supports SFTObjective only."
            )

        dataset = data if isinstance(data, Dataset) else load_sft_jsonl(data)
        for col in self.objective.required_columns():
            if col not in dataset.column_names:
                raise ValueError(f"Dataset missing required column: {col}")

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tokenizer = load_tokenizer(self.model_id)
        model = load_model(self.model_id, self.strategy)
        model = self.strategy.apply(model)
        trainable, total = count_parameters(model)

        def _to_text(example: dict[str, Any]) -> dict[str, str]:
            rendered = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": rendered if isinstance(rendered, str) else str(rendered)}

        train_dataset = dataset.map(_to_text, remove_columns=dataset.column_names)

        train_args = SFTConfig(
            output_dir=str(out),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            seed=self.seed,
            max_length=self.max_seq_length,
            report_to=[],
            packing=False,
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        train_output = trainer.train()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))

        # Trainer leaves gradient checkpointing on; probing needs a generate-safe model.
        from slicktune.recipes.probe import prepare_model_for_inference

        model = prepare_model_for_inference(trainer.model)

        metrics_raw = dict(train_output.metrics)
        tracker = MetricsTracker(out)
        metrics = TrainingMetrics(
            strategy=self.strategy.name,
            objective=self.objective.name,
            model_id=self.model_id,
            train_loss=_as_optional_float(metrics_raw.get("train_loss")),
            train_runtime_sec=_as_optional_float(metrics_raw.get("train_runtime")),
            train_samples_per_second=_as_optional_float(
                metrics_raw.get("train_samples_per_second")
            ),
            trainable_params=trainable,
            total_params=total,
            extras={
                k: v
                for k, v in metrics_raw.items()
                if k
                not in {
                    "train_loss",
                    "train_runtime",
                    "train_samples_per_second",
                }
            },
        )
        tracker.save(metrics)

        return FitResult(
            output_dir=out,
            metrics=metrics,
            model=model,
            tokenizer=tokenizer,
        )


def _as_optional_float(value: Any) -> float | None:
    """Cast a metric value to float when possible.

    Parameters
    ----------
    value : Any
        Raw metric value.

    Returns
    -------
    float or None
        Converted float, or None if ``value`` is None.
    """
    if value is None:
        return None
    return float(value)


__all__ = ["FitResult", "Tuner"]
