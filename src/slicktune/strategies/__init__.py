"""Parameter-update strategies (how weights change)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from slicktune.types import Strategy

BiasType = Literal["none", "all", "lora_only"]


@dataclass(frozen=True)
class LoRAStrategy(Strategy):
    """LoRA PEFT strategy (default production PEFT path).

    Parameters
    ----------
    r : int, optional
        LoRA rank, by default 16.
    alpha : int, optional
        LoRA scaling alpha, by default 32.
    dropout : float, optional
        LoRA dropout probability, by default 0.05.
    target_modules : list[str] | str | None, optional
        Modules to adapt. ``"all-linear"`` lets PEFT discover linear layers,
        by default ``"all-linear"``.
    bias : {"none", "all", "lora_only"}, optional
        Bias training mode passed to PEFT, by default ``"none"``.
    """

    name: str = field(default="lora", init=False)
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str | None = "all-linear"
    bias: BiasType = "none"

    def load_kwargs(self) -> dict[str, Any]:
        """Return empty load kwargs (bf16/fp16 decided by the trainer).

        Returns
        -------
        dict[str, Any]
            Empty mapping; LoRA does not require special load flags.
        """
        return {}

    def apply(self, model: Any) -> Any:
        """Attach LoRA adapters to ``model``.

        Parameters
        ----------
        model : Any
            Hugging Face causal LM.

        Returns
        -------
        Any
            PEFT-wrapped model.
        """
        config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, config)


@dataclass(frozen=True)
class QLoRAStrategy(Strategy):
    """QLoRA strategy: 4-bit quantized base + LoRA adapters.

    Requires CUDA and optional extra ``slick-tune[qlora]`` (bitsandbytes).

    Parameters
    ----------
    r : int, optional
        LoRA rank, by default 16.
    alpha : int, optional
        LoRA scaling alpha, by default 32.
    dropout : float, optional
        LoRA dropout probability, by default 0.05.
    target_modules : list[str] | str | None, optional
        Modules to adapt, by default ``"all-linear"``.
    bias : {"none", "all", "lora_only"}, optional
        Bias training mode, by default ``"none"``.
    """

    name: str = field(default="qlora", init=False)
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str | None = "all-linear"
    bias: BiasType = "none"

    def load_kwargs(self) -> dict[str, Any]:
        """Return bitsandbytes 4-bit quantization config for model load.

        Returns
        -------
        dict[str, Any]
            ``quantization_config`` suitable for ``from_pretrained``.

        Raises
        ------
        ImportError
            If ``bitsandbytes`` is not installed.
        RuntimeError
            If CUDA is not available.
        """
        import torch

        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig as _BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "QLoRA requires bitsandbytes. Install with: uv sync --extra qlora"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA requires a CUDA GPU. On Apple Silicon / CPU use LoRAStrategy."
            )

        bits_and_bytes_config: Any = _BitsAndBytesConfig
        return {
            "quantization_config": bits_and_bytes_config(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        }

    def apply(self, model: Any) -> Any:
        """Prepare a 4-bit model and attach LoRA adapters.

        Parameters
        ----------
        model : Any
            Quantized Hugging Face causal LM.

        Returns
        -------
        Any
            PEFT-wrapped QLoRA model.
        """
        prepare_kbit: Any = prepare_model_for_kbit_training
        model = prepare_kbit(model)
        config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, config)


@dataclass(frozen=True)
class FullStrategy(Strategy):
    """Full fine-tuning: update all model parameters.

    Parameters
    ----------
    None
    """

    name: str = field(default="full", init=False)

    def load_kwargs(self) -> dict[str, Any]:
        """Return empty load kwargs.

        Returns
        -------
        dict[str, Any]
            Empty mapping.
        """
        return {}

    def apply(self, model: Any) -> Any:
        """Enable gradients on all parameters.

        Parameters
        ----------
        model : Any
            Hugging Face causal LM.

        Returns
        -------
        Any
            The same model with ``requires_grad=True`` on all params.
        """
        for param in model.parameters():
            param.requires_grad = True
        return model


__all__ = ["FullStrategy", "LoRAStrategy", "QLoRAStrategy"]
