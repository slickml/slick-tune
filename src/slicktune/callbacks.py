"""Trainer callbacks used by slick-tune strategies."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class AdaLoRACallback(TrainerCallback):
    """Call ``update_and_allocate`` while AdaLoRA grads are still available.

    PEFT requires ``update_and_allocate`` after ``optimizer.step()`` and before
    ``zero_grad()``. Hugging Face ``Trainer`` clears grads before
    ``on_step_end``, so this hooks ``on_optimizer_step`` instead.
    """

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Update AdaLoRA ranks using current parameter gradients.

        Parameters
        ----------
        args : TrainingArguments
            Trainer args (unused).
        state : TrainerState
            Current trainer state (provides ``global_step``).
        control : TrainerControl
            Trainer control flow (unused).
        **kwargs : Any
            Must include ``model`` when available.
        """
        del args, control
        model = kwargs.get("model")
        if model is None:
            return
        update = getattr(model, "update_and_allocate", None)
        if callable(update):
            # Trainer increments global_step after zero_grad; current value is
            # the 0-based step index AdaLoRA expects for this optimizer step.
            update(state.global_step)


__all__ = ["AdaLoRACallback"]
