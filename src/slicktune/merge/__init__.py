"""PEFT multi-adapter loading and TIES / DARE-style merges."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from slicktune.models import load_tokenizer, resolve_dtype

MergeMethod = Literal[
    "ties",
    "ties_svd",
    "dare_ties",
    "dare_linear",
    "dare_ties_svd",
    "dare_linear_svd",
    "linear",
    "svd",
    "cat",
    "magnitude_prune",
    "magnitude_prune_svd",
]

MERGE_METHODS: frozenset[str] = frozenset(
    {
        "ties",
        "ties_svd",
        "dare_ties",
        "dare_linear",
        "dare_ties_svd",
        "dare_linear_svd",
        "linear",
        "svd",
        "cat",
        "magnitude_prune",
        "magnitude_prune_svd",
    }
)

_DENSITY_METHODS: frozenset[str] = frozenset(
    {
        "ties",
        "ties_svd",
        "dare_ties",
        "dare_linear",
        "dare_ties_svd",
        "dare_linear_svd",
        "magnitude_prune",
        "magnitude_prune_svd",
    }
)

# PEFT requires identical LoRA r for these combination types.
_SAME_RANK_METHODS: frozenset[str] = frozenset(
    {
        "linear",
        "ties",
        "dare_ties",
        "dare_linear",
        "magnitude_prune",
    }
)


@dataclass(frozen=True, kw_only=True)
class AdapterRef:
    """Reference to a PEFT adapter directory used in a multi-adapter load or merge.

    Parameters
    ----------
    path : str or Path
        Directory containing ``adapter_config.json`` and adapter weights.
    name : str
        PEFT adapter name used when loading / combining.
    weight : float, optional
        Merge weight for this adapter, by default 1.0.
    """

    path: str | Path
    name: str
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True)
class MergeResult:
    """Artifacts written by :func:`merge_adapters` or :func:`bake_adapter`.

    Parameters
    ----------
    output_dir : Path
        Directory containing the merged adapter or baked full model.
    adapter_name : str or None
        Name of the combined PEFT adapter when ``baked`` is False.
    baked : bool
        Whether adapters were merged into base weights via ``merge_and_unload``.
    """

    output_dir: Path
    adapter_name: str | None
    baked: bool


def parse_adapter_ref(spec: str) -> AdapterRef:
    """Parse a CLI-style adapter spec ``path`` or ``path:weight``.

    Parameters
    ----------
    spec : str
        Adapter directory path, optionally followed by ``:weight``.

    Returns
    -------
    AdapterRef
        Parsed reference with a name derived from the path stem.

    Raises
    ------
    ValueError
        If ``spec`` is empty.
    """
    text = spec.strip()
    if not text:
        raise ValueError("Adapter spec must be a non-empty path or path:weight")

    path_text = text
    weight = 1.0
    if ":" in text:
        left, right = text.rsplit(":", 1)
        try:
            weight = float(right)
            path_text = left
        except ValueError:
            path_text = text
            weight = 1.0

    path = Path(path_text)
    return AdapterRef(path=path, name=_default_adapter_name(path=path), weight=weight)


def load_multi_adapters(
    *,
    model_id: str,
    adapters: list[AdapterRef],
    active: str | None = None,
) -> tuple[Any, PreTrainedTokenizerBase]:
    """Load a base model with one or more PEFT adapters attached.

    Parameters
    ----------
    model_id : str
        Hugging Face id or local path of the base causal LM.
    adapters : list of AdapterRef
        Adapters to attach (at least one). Names must be unique.
    active : str or None, optional
        Adapter name to activate after load. Defaults to the first adapter.

    Returns
    -------
    tuple[Any, PreTrainedTokenizerBase]
        ``(peft_model, tokenizer)``.

    Raises
    ------
    ValueError
        If ``adapters`` is empty, names collide, or an adapter dir is invalid.
    """
    if not adapters:
        raise ValueError("At least one adapter is required")
    _ensure_unique_names(adapters=adapters)
    for ref in adapters:
        _validate_adapter_dir(path=Path(ref.path))

    tokenizer = load_tokenizer(model_id)
    base = _load_base_model(model_id=model_id)

    from peft import PeftModel

    first = adapters[0]
    model = PeftModel.from_pretrained(
        base,
        str(Path(first.path)),
        adapter_name=first.name,
    )
    for ref in adapters[1:]:
        model.load_adapter(str(Path(ref.path)), adapter_name=ref.name)

    model.set_adapter(active if active is not None else first.name)
    model = _maybe_to_device(model)
    return model, tokenizer


def merge_adapters(
    *,
    model_id: str,
    adapters: list[AdapterRef],
    output_dir: str | Path,
    method: MergeMethod | str = "ties",
    density: float | None = 0.5,
    bake: bool = False,
    combined_name: str = "merged",
) -> MergeResult:
    """Combine PEFT adapters with a PEFT weighted-merge method and save the result.

    Parameters
    ----------
    model_id : str
        Base model id used when the adapters were trained.
    adapters : list of AdapterRef
        Adapters to combine (at least two for a true merge; one is allowed and
        simply selects / optionally bakes that adapter).
    output_dir : str or Path
        Destination directory for the combined adapter or baked model.
    method : str, optional
        PEFT ``combination_type`` (e.g. ``ties``, ``dare_ties``, ``linear``),
        by default ``ties``.
    density : float or None, optional
        Prune density in ``[0, 1]`` for TIES / DARE / magnitude methods.
        Ignored for methods that do not use density. Default 0.5.
    bake : bool, optional
        If True, call ``merge_and_unload`` and save a full HF checkpoint.
    combined_name : str, optional
        Name of the new weighted adapter, by default ``merged``.

    Returns
    -------
    MergeResult
        Paths and bake flag for the written artifacts.

    Raises
    ------
    ValueError
        If inputs are invalid, ``method`` is unknown, or adapters have
        incompatible LoRA ranks for ``method``.
    """
    if not adapters:
        raise ValueError("At least one adapter is required")
    method_name = _validate_method(method=method)
    _ensure_unique_names(adapters=adapters)
    if combined_name in {ref.name for ref in adapters}:
        raise ValueError(f"combined_name={combined_name!r} collides with an input adapter name")
    for ref in adapters:
        _validate_adapter_dir(path=Path(ref.path))
    _ensure_compatible_ranks(adapters=adapters, method=method_name)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_multi_adapters(model_id=model_id, adapters=adapters)
    names = [ref.name for ref in adapters]
    weights = [ref.weight for ref in adapters]

    kwargs: dict[str, Any] = {
        "adapters": names,
        "weights": weights,
        "adapter_name": combined_name,
        "combination_type": method_name,
    }
    if method_name in _DENSITY_METHODS and density is not None:
        kwargs["density"] = density

    model.add_weighted_adapter(**kwargs)
    model.set_adapter(combined_name)

    if bake:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        return MergeResult(output_dir=out, adapter_name=None, baked=True)

    _save_peft_adapter(model=model, output_dir=out, adapter_name=combined_name)
    tokenizer.save_pretrained(str(out))
    return MergeResult(output_dir=out, adapter_name=combined_name, baked=False)


def bake_adapter(
    *,
    adapter_dir: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
) -> MergeResult:
    """Merge a single PEFT adapter into base weights for serving.

    Parameters
    ----------
    adapter_dir : str or Path
        Directory from :meth:`Tuner.fit` containing adapter weights.
    output_dir : str or Path
        Destination for the full merged Hugging Face checkpoint.
    model_id : str or None, optional
        Base model id. When omitted, read from ``adapter_config.json``.

    Returns
    -------
    MergeResult
        Baked checkpoint metadata.

    Raises
    ------
    ValueError
        If the adapter directory is invalid or ``base_model_name_or_path`` is
        missing when ``model_id`` is not provided.
    FileNotFoundError
        If ``adapter_config.json`` is missing.
    """
    import json

    path = Path(adapter_dir)
    _validate_adapter_dir(path=path)
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_id = model_id or config.get("base_model_name_or_path")
    if not base_id:
        raise ValueError(
            "model_id is required when adapter_config.json has no base_model_name_or_path"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(str(base_id))
    base = _load_base_model(model_id=str(base_id))

    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(path))
    model = _maybe_to_device(model)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    return MergeResult(output_dir=out, adapter_name=None, baked=True)


def _default_adapter_name(*, path: Path) -> str:
    """Derive a PEFT adapter name from a directory path."""
    stem = path.resolve().name if path.name else "adapter"
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    return cleaned or "adapter"


def _ensure_unique_names(*, adapters: list[AdapterRef]) -> None:
    """Raise if adapter names are not unique."""
    names = [ref.name for ref in adapters]
    if len(names) != len(set(names)):
        raise ValueError(f"Adapter names must be unique; got {names}")


def _validate_adapter_dir(*, path: Path) -> None:
    """Ensure ``path`` looks like a PEFT adapter directory."""
    if not path.is_dir():
        raise ValueError(f"Adapter path is not a directory: {path}")
    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"Missing adapter_config.json in {path}")


def _read_adapter_rank(*, path: Path) -> int | None:
    """Return LoRA ``r`` from ``adapter_config.json``, or None if absent."""
    import json

    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    rank = config.get("r")
    if rank is None:
        return None
    return int(rank)


def _ensure_compatible_ranks(*, adapters: list[AdapterRef], method: str) -> None:
    """Require matching LoRA ranks for PEFT combination types that need them."""
    if method not in _SAME_RANK_METHODS or len(adapters) < 2:
        return
    ranks: dict[str, int | None] = {
        ref.name: _read_adapter_rank(path=Path(ref.path)) for ref in adapters
    }
    present = {name: rank for name, rank in ranks.items() if rank is not None}
    if len(present) < 2:
        return
    unique = set(present.values())
    if len(unique) > 1:
        detail = ", ".join(f"{name}=r{rank}" for name, rank in sorted(present.items()))
        raise ValueError(
            f"Method {method!r} requires all adapters to share the same LoRA r; "
            f"got {detail}. Retrain with matching r, or use an SVD combination "
            f"type (e.g. ties_svd / dare_ties_svd)."
        )


def _validate_method(*, method: str) -> str:
    """Normalize and validate a PEFT combination type."""
    name = method.strip().lower()
    if name not in MERGE_METHODS:
        allowed = ", ".join(sorted(MERGE_METHODS))
        raise ValueError(f"Unknown merge method {method!r}; expected one of: {allowed}")
    return name


def _load_base_model(*, model_id: str) -> Any:
    """Load a causal LM for adapter attachment."""
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": resolve_dtype(),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        model = _maybe_to_device(model)
    return model


def _maybe_to_device(model: Any) -> Any:
    """Move ``model`` to MPS/CUDA when not already device-mapped."""
    if getattr(model, "hf_device_map", None) is not None:
        return model
    if torch.backends.mps.is_available():
        return model.to("mps")
    if torch.cuda.is_available():
        return model.to("cuda")
    return model


def _save_peft_adapter(*, model: Any, output_dir: Path, adapter_name: str) -> None:
    """Save only the combined PEFT adapter to ``output_dir``.

    PEFT writes non-``default`` adapters under ``output_dir / adapter_name``.
    Flatten that layout so :func:`~slicktune.recipes.load_trained` finds
    ``adapter_config.json`` at the checkpoint root (same as ``Tuner.fit``).
    """
    save = getattr(model, "save_pretrained", None)
    if save is None:  # pragma: no cover
        raise TypeError("Model does not support save_pretrained")
    try:
        save(str(output_dir), selected_adapters=[adapter_name])
    except TypeError:
        save(str(output_dir))
    _flatten_adapter_save(output_dir=output_dir, adapter_name=adapter_name)


def _flatten_adapter_save(*, output_dir: Path, adapter_name: str) -> None:
    """Move ``output_dir / adapter_name`` adapter files up to ``output_dir``."""
    nested = output_dir / adapter_name
    if not (nested / "adapter_config.json").is_file():
        return
    if (output_dir / "adapter_config.json").is_file():
        return
    for item in nested.iterdir():
        target = output_dir / item.name
        if target.exists():
            continue
        item.rename(target)
    # Remove empty nested dir (ignore leftovers that could not be moved).
    with contextlib.suppress(OSError):
        nested.rmdir()


__all__ = [
    "MERGE_METHODS",
    "AdapterRef",
    "MergeMethod",
    "MergeResult",
    "bake_adapter",
    "load_multi_adapters",
    "merge_adapters",
    "parse_adapter_ref",
]
