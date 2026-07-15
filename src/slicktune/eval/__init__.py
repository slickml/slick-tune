"""Phase-2 evaluation: holdout perplexity and judges."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from slicktune.data import load_probe_jsonl, load_sft_jsonl
from slicktune.recipes.probe import generate_reply, prepare_model_for_inference


@dataclass(kw_only=True)
class HoldoutEvalResult:
    """Holdout negative-log-likelihood / perplexity summary.

    Parameters
    ----------
    eval_loss : float
        Mean token NLL on the holdout set.
    perplexity : float
        ``exp(eval_loss)``.
    num_examples : int
        Number of evaluated examples.
    """

    eval_loss: float
    perplexity: float
    num_examples: int


@dataclass(kw_only=True)
class JudgeResult:
    """Outcome of judging one generation.

    Parameters
    ----------
    prompt : str
        User prompt.
    generation : str
        Model completion.
    score : float
        Score in ``[0, 1]``.
    rationale : str
        Short explanation from the judge.
    """

    prompt: str
    generation: str
    score: float
    rationale: str


@dataclass(kw_only=True)
class JudgeReport:
    """Aggregate judge outcomes.

    Parameters
    ----------
    results : list[JudgeResult]
        Per-example judgments.
    """

    results: list[JudgeResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        """Return mean score across judged examples.

        Returns
        -------
        float
            Mean in ``[0, 1]``, or ``0.0`` when empty.
        """
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


@dataclass(frozen=True, kw_only=True)
class Judge(ABC):
    """Score model generations for quality / correctness."""

    @abstractmethod
    def judge(self, *, prompt: str, generation: str, **context: Any) -> JudgeResult:
        """Score a single generation.

        Parameters
        ----------
        prompt : str
            User prompt.
        generation : str
            Model completion.
        **context : Any
            Judge-specific extras (e.g. ``must_contain``).

        Returns
        -------
        JudgeResult
            Score and rationale.
        """


@dataclass(frozen=True, kw_only=True)
class SubstringJudge(Judge):
    """Deterministic judge: pass if ``must_contain`` appears in the generation."""

    case_sensitive: bool = False

    def judge(self, *, prompt: str, generation: str, **context: Any) -> JudgeResult:
        """Score via substring match.

        Parameters
        ----------
        prompt : str
            User prompt.
        generation : str
            Model completion.
        **context : Any
            Must include ``must_contain``.

        Returns
        -------
        JudgeResult
            Score ``1.0`` or ``0.0``.
        """
        needle = str(context.get("must_contain", ""))
        hay = generation if self.case_sensitive else generation.lower()
        needle_cmp = needle if self.case_sensitive else needle.lower()
        passed = bool(needle_cmp) and needle_cmp in hay
        return JudgeResult(
            prompt=prompt,
            generation=generation,
            score=1.0 if passed else 0.0,
            rationale="substring match" if passed else "substring missing",
        )


_JUDGE_SYSTEM_PROMPT = "You are a strict evaluation judge. Reply with a single integer score only."


def _single_token_ids_for_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Return tokenizer ids that decode exactly to ``text`` as one token.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Judge tokenizer.
    text : str
        Target string (e.g. ``\"8\"`` or ``\" 8\"``).

    Returns
    -------
    list[int]
        Matching token ids (possibly empty).
    """
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) == 1:
        return [encoded[0]]
    return []


def _generate_score_0_to_10(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    rubric: str,
) -> str:
    """Generate a 0–10 score with decoding constrained to digit tokens.

    Uses a plain completion prompt ending in ``SCORE:`` (not a chat turn) so
    the next tokens continue the score. Restricts generation to digit tokens so
    small models cannot answer ``Yes`` / ``True`` instead.

    Parameters
    ----------
    model : Any
        Causal LM.
    tokenizer : PreTrainedTokenizerBase
        Matching tokenizer.
    rubric : str
        Judge prompt that already ends with ``SCORE:``.

    Returns
    -------
    str
        Decoded score text (typically a single integer).
    """
    prepare_model_for_inference(model)
    # Plain completion — chat templates start a new assistant turn and drop the
    # ``SCORE:`` prefix continuity that small judges need.
    text = rubric if rubric.endswith("SCORE:") else f"{rubric}\nSCORE:"
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    digit_ids: dict[int, int] = {}
    for value in range(10):
        for candidate in (str(value), f" {value}"):
            for token_id in _single_token_ids_for_text(tokenizer, candidate):
                digit_ids[value] = token_id
    one_ids = {
        token_id
        for candidate in ("1", " 1")
        for token_id in _single_token_ids_for_text(tokenizer, candidate)
    }
    zero_ids = {
        token_id
        for candidate in ("0", " 0")
        for token_id in _single_token_ids_for_text(tokenizer, candidate)
    }
    allowed_first = sorted(set(digit_ids.values()) | one_ids)
    if not allowed_first:
        # Tokenizer edge case: fall back to unconstrained short generation.
        return generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=rubric,
            max_new_tokens=4,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
        )

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    def _allow_score_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = input_ids[prompt_len:]
        if generated.numel() == 0:
            return allowed_first
        if generated.numel() == 1 and int(generated[0]) in one_ids and zero_ids:
            # Allow completing "10".
            allowed = list(zero_ids)
            if eos_id is not None:
                allowed.append(int(eos_id))
            return allowed
        return [int(eos_id)] if eos_id is not None else allowed_first

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
            prefix_allowed_tokens_fn=_allow_score_tokens,
        )

    new_tokens = output_ids[0][prompt_len:]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


@dataclass(frozen=True, kw_only=True)
class LLMJudge(Judge):
    """Ask an LLM to score a completion from 0–10, normalized to ``[0, 1]``.

    Uses digit-constrained decoding so small models cannot wander into
    ``Yes`` / ``True`` instead of a numeric score.

    Parameters
    ----------
    model : Any
        Causal LM used as the judge (often the same trained model).
    tokenizer : PreTrainedTokenizerBase
        Matching tokenizer.
    max_new_tokens : int, optional
        Unused for constrained scoring (kept for API compatibility), by default 2.
    """

    model: Any
    tokenizer: PreTrainedTokenizerBase
    max_new_tokens: int = 2

    def judge(self, *, prompt: str, generation: str, **context: Any) -> JudgeResult:
        """Score via an LLM rubric prompt.

        Parameters
        ----------
        prompt : str
            User prompt.
        generation : str
            Model completion.
        **context : Any
            Optional ``criteria`` string.

        Returns
        -------
        JudgeResult
            Normalized score parsed from the judge reply.
        """
        criteria = str(context.get("criteria", "factual accuracy and relevance"))
        rubric = (
            f"Rate the ASSISTANT reply for {criteria}.\n"
            "Reply with one integer after SCORE.\n\n"
            "USER: Who founded SlickML?\n"
            "ASSISTANT: Amirhessam Tahmassebi is the founder of SlickML.\n"
            "SCORE: 9\n\n"
            "USER: Who founded SlickML?\n"
            "ASSISTANT: I have no idea.\n"
            "SCORE: 1\n\n"
            f"USER: {prompt}\n"
            f"ASSISTANT: {generation}\n"
            "SCORE:"
        )
        # max_new_tokens is retained for API compatibility; scoring uses a
        # fixed 2-token constrained decode.
        _ = self.max_new_tokens
        raw = _generate_score_0_to_10(
            model=self.model,
            tokenizer=self.tokenizer,
            rubric=rubric,
        )
        score = _parse_score_0_to_10(raw)
        return JudgeResult(
            prompt=prompt,
            generation=generation,
            score=score,
            rationale=raw.strip()[:200],
        )


def _parse_score_0_to_10(text: str) -> float:
    """Parse a 0–10 integer from judge output and normalize to ``[0, 1]``.

    Ignores scale echoes such as ``0-10`` / ``0 to 10`` so the leading zero is
    not mistaken for the score. Prefers an explicit ``SCORE: N`` or the last
    remaining integer in ``[0, 10]``.

    Parameters
    ----------
    text : str
        Judge model output.

    Returns
    -------
    float
        Normalized score; ``0.0`` if no usable integer is found.
    """
    cleaned = re.sub(r"\b0\s*(?:-|–|to)\s*10\b", " ", text, flags=re.IGNORECASE)
    for pattern in (
        r"(?i)\bscore\s*[:=]?\s*(10|[0-9])\b",
        r"(?m)^\s*(10|[0-9])\s*[./]?\s*(?:10)?\s*$",
        r"\b(10|[0-9])\b",
    ):
        matches = list(re.finditer(pattern, cleaned))
        if matches:
            return float(matches[-1].group(1)) / 10.0
    return 0.0


def compute_holdout_perplexity(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    data: str | Path | Dataset,
    max_length: int = 512,
) -> HoldoutEvalResult:
    """Compute mean NLL and perplexity on a holdout SFT JSONL / dataset.

    Parameters
    ----------
    model : Any
        Causal LM (base or PEFT).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer.
    data : str or Path or Dataset
        Holdout SFT data with ``messages`` (or loadable JSONL).
    max_length : int, optional
        Truncation length, by default 512.

    Returns
    -------
    HoldoutEvalResult
        Loss / perplexity summary.

    Raises
    ------
    ValueError
        If no examples are available.
    """
    dataset = data if isinstance(data, Dataset) else load_sft_jsonl(data)
    if len(dataset) == 0:
        raise ValueError("Holdout dataset is empty")

    prepare_model_for_inference(model)
    device = next(model.parameters()).device
    losses: list[float] = []

    for row in dataset:
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = rendered if isinstance(rendered, str) else str(rendered)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[-1] < 2:
            continue
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, labels=input_ids)
        loss = float(outputs.loss.detach().cpu())
        if math.isfinite(loss):
            losses.append(loss)

    if not losses:
        raise ValueError("No holdout examples produced a finite loss")

    mean_loss = sum(losses) / len(losses)
    return HoldoutEvalResult(
        eval_loss=mean_loss,
        perplexity=math.exp(mean_loss),
        num_examples=len(losses),
    )


def run_judge_on_probes(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    probe_path: str | Path,
    judge: Judge,
    max_new_tokens: int = 128,
) -> JudgeReport:
    """Generate replies for probe prompts and score them with ``judge``.

    Parameters
    ----------
    model : Any
        Model under evaluation.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer.
    probe_path : str or Path
        Probe JSONL (``prompt``, ``must_contain``).
    judge : Judge
        Scoring strategy.
    max_new_tokens : int, optional
        Generation length, by default 128.

    Returns
    -------
    JudgeReport
        Aggregate scores.
    """
    prepare_model_for_inference(model)
    results: list[JudgeResult] = []
    for row in load_probe_jsonl(probe_path):
        generation = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=row["prompt"],
            max_new_tokens=max_new_tokens,
        )
        results.append(
            judge.judge(
                prompt=row["prompt"],
                generation=generation,
                must_contain=row["must_contain"],
            )
        )
    return JudgeReport(results=results)


__all__ = [
    "HoldoutEvalResult",
    "Judge",
    "JudgeReport",
    "JudgeResult",
    "LLMJudge",
    "SubstringJudge",
    "compute_holdout_perplexity",
    "run_judge_on_probes",
]
