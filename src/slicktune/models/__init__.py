"""Model and tokenizer loading helpers."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from slicktune.types import Strategy


def resolve_dtype() -> torch.dtype:
    """Pick a default compute dtype for the current device.

    Returns
    -------
    torch.dtype
        ``bfloat16`` on CUDA when supported, else ``float32`` (including MPS,
        which is more stable for small SFT smoke tests than float16).
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer and ensure a pad token exists.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or local path.

    Returns
    -------
    PreTrainedTokenizerBase
        Tokenizer with ``pad_token`` set when missing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_id: str, strategy: Strategy) -> Any:
    """Load a causal LM configured for ``strategy``.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or local path.
    strategy : Strategy
        Parameter-update strategy providing load kwargs.

    Returns
    -------
    Any
        Hugging Face causal LM (not yet PEFT-wrapped unless strategy does so
        during apply).
    """
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        **strategy.load_kwargs(),
    }
    if "quantization_config" not in kwargs:
        kwargs["dtype"] = resolve_dtype()
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        if torch.backends.mps.is_available():
            model = model.to("mps")  # type: ignore[arg-type]
        elif torch.cuda.is_available():
            model = model.to("cuda")  # type: ignore[arg-type]
    return model


__all__ = ["load_model", "load_tokenizer", "resolve_dtype"]
