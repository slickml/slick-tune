"""Post-train probing to verify the model absorbed fine-tuning data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from slicktune.data import DEFAULT_SYSTEM_PROMPT, load_probe_jsonl
from slicktune.models import resolve_dtype


@dataclass(kw_only=True)
class ProbeResult:
    """Outcome of a single probe question.

    Parameters
    ----------
    prompt : str
        User question.
    must_contain : str
        Expected substring (case-insensitive).
    generation : str
        Model completion.
    passed : bool
        Whether ``must_contain`` appears in ``generation``.
    """

    prompt: str
    must_contain: str
    generation: str
    passed: bool


@dataclass(kw_only=True)
class ProbeReport:
    """Aggregate probe results.

    Parameters
    ----------
    results : list[ProbeResult]
        Per-question outcomes.
    """

    results: list[ProbeResult]

    @property
    def pass_rate(self) -> float:
        """Return fraction of probes that passed.

        Returns
        -------
        float
            Pass rate in ``[0, 1]``.
        """
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)


def prepare_model_for_inference(model: Any) -> Any:
    """Switch a post-training model into a generate-safe state.

    Disables gradient checkpointing (incompatible with KV cache) and enables
    ``use_cache`` so decoding does not collapse into token loops.

    Parameters
    ----------
    model : Any
        Trained base or PEFT model.

    Returns
    -------
    Any
        The same model, mutated for inference.
    """
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        base = get_base()
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        if hasattr(base, "config"):
            base.config.use_cache = True
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model


def generate_reply(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate an assistant reply for ``prompt``.

    Parameters
    ----------
    model : Any
        Causal LM (base or PEFT).
    tokenizer : PreTrainedTokenizerBase
        Matching tokenizer.
    prompt : str
        User message.
    max_new_tokens : int, optional
        Generation length, by default 128.
    system_prompt : str, optional
        System message for chat-template models, by default the about-me
        prompt used for personal probes.

    Returns
    -------
    str
        Decoded assistant text.
    """
    prepare_model_for_inference(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if isinstance(decoded, list):
        return " ".join(str(part) for part in decoded).strip()
    return str(decoded).strip()


def run_probes(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    probe_path: str | Path,
    max_new_tokens: int = 128,
) -> ProbeReport:
    """Run probe questions against a model.

    Parameters
    ----------
    model : Any
        Trained model.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer.
    probe_path : str or Path
        JSONL probe file.
    max_new_tokens : int, optional
        Generation length, by default 128.

    Returns
    -------
    ProbeReport
        Aggregate probe outcomes.
    """
    prepare_model_for_inference(model)
    results: list[ProbeResult] = []
    for row in load_probe_jsonl(probe_path):
        generation = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=row["prompt"],
            max_new_tokens=max_new_tokens,
        )
        passed = row["must_contain"].lower() in generation.lower()
        results.append(
            ProbeResult(
                prompt=row["prompt"],
                must_contain=row["must_contain"],
                generation=generation,
                passed=passed,
            )
        )
    return ProbeReport(results=results)


def load_trained(output_dir: str | Path) -> tuple[Any, PreTrainedTokenizerBase]:
    """Load a saved adapter/model directory for probing.

    Parameters
    ----------
    output_dir : str or Path
        Directory produced by :meth:`Tuner.fit`.

    Returns
    -------
    tuple[Any, PreTrainedTokenizerBase]
        ``(model, tokenizer)``.
    """
    path = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype()
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)

    if torch.backends.mps.is_available():
        model = model.to("mps")
    elif torch.cuda.is_available():
        model = model.to("cuda")

    return prepare_model_for_inference(model), tokenizer


__all__ = [
    "ProbeReport",
    "ProbeResult",
    "generate_reply",
    "load_trained",
    "prepare_model_for_inference",
    "run_probes",
]
