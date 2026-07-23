"""High-level Tuner API composing model × strategy × objective × data × metrics."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, KTOConfig, KTOTrainer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from slicktune.callbacks import AdaLoRACallback
from slicktune.data import (
    load_grpo_jsonl,
    load_kto_jsonl,
    load_preference_jsonl,
    load_sft_jsonl,
)
from slicktune.eval import (
    Judge,
    SubstringJudge,
    compute_holdout_perplexity,
    run_judge_on_probes,
)
from slicktune.metrics import MetricsTracker, TrainingMetrics, count_parameters
from slicktune.models import load_model, load_tokenizer
from slicktune.objectives import (
    DPOObjective,
    GRPOObjective,
    KTOObjective,
    ORPOObjective,
    SFTObjective,
)
from slicktune.rewards import substring_must_contain_reward
from slicktune.strategies import AdaLoRAStrategy, FullStrategy
from slicktune.types import Objective, Strategy

# Silence TRL experimental import noise for ORPO unless the user opts out.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class Tuner:
    """Composable fine-tuning entry point.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or local path.
    strategy : Strategy
        Parameter-update strategy (LoRA, DoRA, AdaLoRA, QLoRA, full, ...).
    objective : Objective
        Training objective (SFT, DPO, ORPO, KTO, or GRPO).
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
    eval_data : str or Path or Dataset or None, optional
        Optional holdout SFT JSONL/dataset for perplexity after fit.
    probe_path : str or Path or None, optional
        Optional probe JSONL judged after fit (substring or custom judge).
    judge : Judge or None, optional
        Judge used with ``probe_path``; defaults to :class:`SubstringJudge`.
    adapter_path : str or Path or None, optional
        Optional PEFT adapter directory to warm-start from (skip fresh
        ``strategy.apply``). Useful for GRPO after a short SFT run.
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
    eval_data: str | Path | Dataset | None = None
    probe_path: str | Path | None = None
    judge: Judge | None = None
    adapter_path: str | Path | None = None

    def fit(self, data: str | Path | Dataset) -> FitResult:
        """Run fine-tuning for the configured objective.

        Parameters
        ----------
        data : str or Path or Dataset
            Path to objective-specific JSONL, or an in-memory dataset.

        Returns
        -------
        FitResult
            Trained artifacts and metrics.

        Raises
        ------
        TypeError
            If the objective is not supported.
        ValueError
            If required dataset columns are missing.
        """
        if isinstance(self.objective, SFTObjective):
            return self._fit_sft(data)
        if isinstance(self.objective, DPOObjective):
            return self._fit_dpo(data)
        if isinstance(self.objective, ORPOObjective):
            return self._fit_orpo(data)
        if isinstance(self.objective, KTOObjective):
            return self._fit_kto(data)
        if isinstance(self.objective, GRPOObjective):
            return self._fit_grpo(data)
        raise TypeError(
            f"Objective '{self.objective.name}' is not implemented. "
            "Supported: sft, dpo, orpo, kto, grpo."
        )

    def _load_dataset(
        self,
        data: str | Path | Dataset,
        *,
        loader: Any,
    ) -> Dataset:
        """Load or validate a dataset for the active objective.

        Parameters
        ----------
        data : str or Path or Dataset
            Path or in-memory dataset.
        loader : callable
            JSONL loader used when ``data`` is a path.

        Returns
        -------
        Dataset
            Dataset with required columns present.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        dataset = data if isinstance(data, Dataset) else loader(data)
        for col in self.objective.required_columns():
            if col not in dataset.column_names:
                raise ValueError(f"Dataset missing required column: {col}")
        return dataset

    def _prepare_model(
        self,
        *,
        num_examples: int,
    ) -> tuple[Strategy, Any, PreTrainedTokenizerBase, int, int]:
        """Load tokenizer/model, apply strategy, and count parameters.

        Parameters
        ----------
        num_examples : int
            Training set size (for AdaLoRA schedule estimation).

        Returns
        -------
        tuple
            ``(strategy, model, tokenizer, trainable, total)``.
        """
        strategy = self._prepare_strategy(num_examples)
        tokenizer = load_tokenizer(self.model_id)
        model = load_model(model_id=self.model_id, strategy=strategy)
        if self.adapter_path is not None:
            from peft import PeftModel

            # Default is_trainable=False freezes adapter weights (no grad_fn).
            model = PeftModel.from_pretrained(
                model,
                str(self.adapter_path),
                is_trainable=True,
            )
            model.train()
        else:
            model = strategy.apply(model)
        trainable, total = count_parameters(model)
        return strategy, model, tokenizer, trainable, total

    def _callbacks_for(self, strategy: Strategy) -> list[TrainerCallback]:
        """Build trainer callbacks for ``strategy``.

        Parameters
        ----------
        strategy : Strategy
            Active parameter strategy.

        Returns
        -------
        list[TrainerCallback]
            Callbacks (AdaLoRA when applicable).
        """
        callbacks: list[TrainerCallback] = []
        if isinstance(strategy, AdaLoRAStrategy):
            callbacks.append(AdaLoRACallback())
        return callbacks

    def _ref_model_for_preference(self, *, strategy: Strategy, model: Any) -> Any | None:
        """Choose a DPO/KTO reference model.

        PEFT runs omit ``ref_model`` so TRL uses adapter disable. Full FT loads a
        frozen copy of the base checkpoint.

        Parameters
        ----------
        strategy : Strategy
            Active strategy.
        model : Any
            Policy model (unused for PEFT; documented for symmetry).

        Returns
        -------
        Any or None
            Frozen reference model, or None for PEFT.
        """
        del model  # PEFT path ignores the policy instance for ref construction.
        if isinstance(strategy, FullStrategy):
            ref = load_model(model_id=self.model_id, strategy=strategy)
            ref.eval()
            for param in ref.parameters():
                param.requires_grad = False
            return ref
        return None

    def _fit_sft(self, data: str | Path | Dataset) -> FitResult:
        """Run supervised fine-tuning via TRL :class:`~trl.SFTTrainer`."""
        dataset = self._load_dataset(data, loader=load_sft_jsonl)
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        strategy, model, tokenizer, trainable, total = self._prepare_model(
            num_examples=len(dataset)
        )

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
            callbacks=self._callbacks_for(strategy),
        )
        train_output = trainer.train()
        return self._finalize_fit(
            out=out,
            trainer=trainer,
            tokenizer=tokenizer,
            strategy=strategy,
            trainable=trainable,
            total=total,
            train_output=train_output,
        )

    def _fit_dpo(self, data: str | Path | Dataset) -> FitResult:
        """Run DPO via TRL :class:`~trl.DPOTrainer`."""
        assert isinstance(self.objective, DPOObjective)
        dataset = self._load_dataset(data, loader=load_preference_jsonl)
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        strategy, model, tokenizer, trainable, total = self._prepare_model(
            num_examples=len(dataset)
        )
        train_args = DPOConfig(
            output_dir=str(out),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            seed=self.seed,
            max_length=self.max_seq_length,
            beta=self.objective.beta,
            loss_type=[self.objective.loss_type],
            report_to=[],
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=self._ref_model_for_preference(strategy=strategy, model=model),
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=self._callbacks_for(strategy),
        )
        train_output = trainer.train()
        return self._finalize_fit(
            out=out,
            trainer=trainer,
            tokenizer=tokenizer,
            strategy=strategy,
            trainable=trainable,
            total=total,
            train_output=train_output,
        )

    def _fit_orpo(self, data: str | Path | Dataset) -> FitResult:
        """Run ORPO via TRL experimental :class:`~trl.experimental.orpo.ORPOTrainer`."""
        assert isinstance(self.objective, ORPOObjective)
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        dataset = self._load_dataset(data, loader=load_preference_jsonl)
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        strategy, model, tokenizer, trainable, total = self._prepare_model(
            num_examples=len(dataset)
        )
        train_args = ORPOConfig(
            output_dir=str(out),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            seed=self.seed,
            max_length=self.max_seq_length,
            beta=self.objective.beta,
            report_to=[],
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
        )
        trainer = ORPOTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=self._callbacks_for(strategy),
        )
        train_output = trainer.train()
        return self._finalize_fit(
            out=out,
            trainer=trainer,
            tokenizer=tokenizer,
            strategy=strategy,
            trainable=trainable,
            total=total,
            train_output=train_output,
        )

    def _fit_kto(self, data: str | Path | Dataset) -> FitResult:
        """Run KTO via TRL :class:`~trl.KTOTrainer`."""
        assert isinstance(self.objective, KTOObjective)
        dataset = self._load_dataset(data, loader=load_kto_jsonl)
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        strategy, model, tokenizer, trainable, total = self._prepare_model(
            num_examples=len(dataset)
        )
        # TRL requires per-device batch size > 1 so the KL term is meaningful.
        batch_size = max(2, self.per_device_train_batch_size)
        train_args = KTOConfig(
            output_dir=str(out),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            seed=self.seed,
            max_length=self.max_seq_length,
            beta=self.objective.beta,
            desirable_weight=self.objective.desirable_weight,
            undesirable_weight=self.objective.undesirable_weight,
            report_to=[],
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
        )
        trainer = KTOTrainer(
            model=model,
            ref_model=self._ref_model_for_preference(strategy=strategy, model=model),
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=self._callbacks_for(strategy),
        )
        train_output = trainer.train()
        return self._finalize_fit(
            out=out,
            trainer=trainer,
            tokenizer=tokenizer,
            strategy=strategy,
            trainable=trainable,
            total=total,
            train_output=train_output,
        )

    def _fit_grpo(self, data: str | Path | Dataset) -> FitResult:
        """Run GRPO via TRL :class:`~trl.GRPOTrainer` with substring rewards."""
        assert isinstance(self.objective, GRPOObjective)
        dataset = self._load_dataset(data, loader=load_grpo_jsonl)
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        strategy, model, tokenizer, trainable, total = self._prepare_model(
            num_examples=len(dataset)
        )
        num_gens = max(2, self.objective.num_generations)
        batch_size = max(1, self.per_device_train_batch_size)
        grad_accum = max(1, self.gradient_accumulation_steps)
        # generation_batch_size = batch * processes * steps_per_generation
        # (steps_per_generation defaults to gradient_accumulation_steps).
        while (batch_size * grad_accum) % num_gens != 0:
            grad_accum += 1

        train_args = GRPOConfig(
            output_dir=str(out),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            seed=self.seed,
            num_generations=num_gens,
            max_completion_length=self.objective.max_completion_length,
            temperature=self.objective.temperature,
            beta=self.objective.beta,
            report_to=[],
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=substring_must_contain_reward,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=self._callbacks_for(strategy),
        )
        train_output = trainer.train()
        return self._finalize_fit(
            out=out,
            trainer=trainer,
            tokenizer=tokenizer,
            strategy=strategy,
            trainable=trainable,
            total=total,
            train_output=train_output,
        )

    def _finalize_fit(
        self,
        *,
        out: Path,
        trainer: Any,
        tokenizer: PreTrainedTokenizerBase,
        strategy: Strategy,
        trainable: int,
        total: int,
        train_output: Any,
    ) -> FitResult:
        """Save artifacts, run optional eval/probes, and write metrics."""
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))

        from slicktune.recipes.probe import prepare_model_for_inference

        model = prepare_model_for_inference(trainer.model)

        metrics_raw = dict(train_output.metrics)
        eval_loss: float | None = _as_optional_float(metrics_raw.get("eval_loss"))
        eval_perplexity: float | None = None
        judge_score: float | None = None
        probe_pass_rate: float | None = None

        if self.eval_data is not None:
            holdout = compute_holdout_perplexity(
                model=model,
                tokenizer=tokenizer,
                data=self.eval_data,
                max_length=self.max_seq_length,
            )
            eval_loss = holdout.eval_loss
            eval_perplexity = holdout.perplexity

        if self.probe_path is not None:
            active_judge = self.judge if self.judge is not None else SubstringJudge()
            report = run_judge_on_probes(
                model=model,
                tokenizer=tokenizer,
                probe_path=self.probe_path,
                judge=active_judge,
            )
            judge_score = report.mean_score
            if isinstance(active_judge, SubstringJudge):
                probe_pass_rate = judge_score

        tracker = MetricsTracker(output_dir=out)
        metrics = TrainingMetrics(
            strategy=strategy.name,
            objective=self.objective.name,
            model_id=self.model_id,
            train_loss=_as_optional_float(metrics_raw.get("train_loss")),
            eval_loss=eval_loss,
            train_runtime_sec=_as_optional_float(metrics_raw.get("train_runtime")),
            train_samples_per_second=_as_optional_float(
                metrics_raw.get("train_samples_per_second")
            ),
            trainable_params=trainable,
            total_params=total,
            probe_pass_rate=probe_pass_rate,
            eval_perplexity=eval_perplexity,
            judge_score=judge_score,
            extras={
                k: v
                for k, v in metrics_raw.items()
                if k
                not in {
                    "train_loss",
                    "eval_loss",
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

    def _prepare_strategy(self, num_examples: int) -> Strategy:
        """Adjust AdaLoRA schedule knobs from the run shape when left at defaults.

        When ``total_step`` is still the library default (1000), replace it with
        an estimate from dataset size × epochs. If ``tinit`` and ``tfinal`` are
        both still 0, also set a short warmup / final fine-tune window so rank
        pruning does not start on step 0 (important for tiny SFT sets).

        Parameters
        ----------
        num_examples : int
            Number of training examples.

        Returns
        -------
        Strategy
            Possibly replaced AdaLoRA strategy with estimated schedule knobs.
        """
        strategy = self.strategy
        if not isinstance(strategy, AdaLoRAStrategy):
            return strategy
        steps_per_epoch = max(
            1,
            math.ceil(num_examples / max(1, self.per_device_train_batch_size))
            // max(1, self.gradient_accumulation_steps),
        )
        # Prefer an estimate from the run shape when the user left the default.
        if strategy.total_step != 1000:
            return strategy

        estimated = max(1, int(steps_per_epoch * self.num_train_epochs))
        if strategy.tinit == 0 and strategy.tfinal == 0 and estimated >= 6:
            return replace(
                strategy,
                total_step=estimated,
                tinit=max(1, estimated // 3),
                tfinal=max(1, estimated // 6),
            )
        return replace(strategy, total_step=estimated)


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
