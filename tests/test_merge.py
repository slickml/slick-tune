"""Tests for PEFT multi-adapter load and TIES / DARE merges."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from assertpy import assert_that

from slicktune.merge import (
    MERGE_METHODS,
    AdapterRef,
    MergeResult,
    bake_adapter,
    load_multi_adapters,
    merge_adapters,
    parse_adapter_ref,
)


def _adapter_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "fake/base"}',
        encoding="utf-8",
    )
    return path


def test_parse_adapter_ref_path_only(tmp_path: Path) -> None:
    """Bare path defaults weight to 1.0 and uses directory stem as name."""
    path = tmp_path / "sft_lora"
    ref = parse_adapter_ref(str(path))
    assert_that(ref.path).is_equal_to(path)
    assert_that(ref.name).is_equal_to("sft_lora")
    assert_that(ref.weight).is_equal_to(1.0)


def test_parse_adapter_ref_with_weight(tmp_path: Path) -> None:
    """``path:weight`` splits a trailing float weight."""
    path = tmp_path / "dpo_lora"
    ref = parse_adapter_ref(f"{path}:0.5")
    assert_that(ref.path).is_equal_to(path)
    assert_that(ref.weight).is_equal_to(0.5)
    assert_that(ref.name).is_equal_to("dpo_lora")


def test_parse_adapter_ref_non_float_suffix(tmp_path: Path) -> None:
    """Non-float suffix after ``:`` is treated as part of the path."""
    path = tmp_path / "adapter:v2"
    ref = parse_adapter_ref(str(path))
    assert_that(ref.path).is_equal_to(path)
    assert_that(ref.weight).is_equal_to(1.0)


def test_parse_adapter_ref_empty() -> None:
    """Empty specs raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        parse_adapter_ref("   ")


def test_merge_methods_include_ties_dare() -> None:
    """Documented Phase 5 methods are advertised."""
    assert_that(MERGE_METHODS).contains("ties", "dare_ties", "dare_linear", "linear")


def test_load_multi_adapters_requires_adapters() -> None:
    """Empty adapter lists are rejected."""
    with pytest.raises(ValueError, match="At least one adapter"):
        load_multi_adapters(model_id="fake/base", adapters=[])


def test_load_multi_adapters_missing_config(tmp_path: Path) -> None:
    """Adapter dirs without adapter_config.json raise ValueError."""
    bad = tmp_path / "nope"
    bad.mkdir()
    with pytest.raises(ValueError, match="adapter_config.json"):
        load_multi_adapters(
            model_id="fake/base",
            adapters=[AdapterRef(path=bad, name="a")],
        )


def test_load_multi_adapters_duplicate_names(tmp_path: Path) -> None:
    """Duplicate PEFT adapter names are rejected."""
    a = _adapter_dir(tmp_path, "a")
    b = _adapter_dir(tmp_path, "b")
    with pytest.raises(ValueError, match="unique"):
        load_multi_adapters(
            model_id="fake/base",
            adapters=[
                AdapterRef(path=a, name="same"),
                AdapterRef(path=b, name="same"),
            ],
        )


def test_load_multi_adapters_loads_and_sets_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First adapter uses from_pretrained; others use load_adapter."""
    a = _adapter_dir(tmp_path, "alpha")
    b = _adapter_dir(tmp_path, "beta")
    tok = MagicMock()
    base = MagicMock()
    peft_model = MagicMock()
    peft_model.to.return_value = peft_model

    monkeypatch.setattr("slicktune.merge.load_tokenizer", lambda model_id: tok)
    monkeypatch.setattr("slicktune.merge._load_base_model", lambda *, model_id: base)
    monkeypatch.setattr("slicktune.merge._maybe_to_device", lambda model: model)

    fake_peft = MagicMock()
    fake_peft.from_pretrained.return_value = peft_model
    monkeypatch.setattr(
        "peft.PeftModel",
        fake_peft,
        raising=False,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        SimpleNamespace(PeftModel=fake_peft),
    )

    model, tokenizer = load_multi_adapters(
        model_id="fake/base",
        adapters=[
            AdapterRef(path=a, name="alpha", weight=1.0),
            AdapterRef(path=b, name="beta", weight=0.5),
        ],
        active="beta",
    )

    assert_that(model).is_equal_to(peft_model)
    assert_that(tokenizer).is_equal_to(tok)
    fake_peft.from_pretrained.assert_called_once_with(
        base,
        str(a),
        adapter_name="alpha",
    )
    peft_model.load_adapter.assert_called_once_with(str(b), adapter_name="beta")
    peft_model.set_adapter.assert_called_once_with("beta")


def test_merge_adapters_calls_weighted_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """merge_adapters wires add_weighted_adapter and saves the combined adapter."""
    a = _adapter_dir(tmp_path, "a_lora")
    b = _adapter_dir(tmp_path, "b_lora")
    out = tmp_path / "merged"
    tok = MagicMock()
    peft_model = MagicMock()
    captured: dict[str, Any] = {}

    def _fake_load(
        *,
        model_id: str,
        adapters: list[AdapterRef],
        active: str | None = None,
    ) -> tuple[Any, Any]:
        captured["model_id"] = model_id
        captured["adapters"] = adapters
        captured["active"] = active
        return peft_model, tok

    monkeypatch.setattr("slicktune.merge.load_multi_adapters", _fake_load)

    result = merge_adapters(
        model_id="fake/base",
        adapters=[
            AdapterRef(path=a, name="a_lora", weight=1.0),
            AdapterRef(path=b, name="b_lora", weight=0.5),
        ],
        output_dir=out,
        method="ties",
        density=0.4,
        bake=False,
        combined_name="merged",
    )

    peft_model.add_weighted_adapter.assert_called_once_with(
        adapters=["a_lora", "b_lora"],
        weights=[1.0, 0.5],
        adapter_name="merged",
        combination_type="ties",
        density=0.4,
    )
    peft_model.set_adapter.assert_called_with("merged")
    peft_model.save_pretrained.assert_called()
    tok.save_pretrained.assert_called_once_with(str(out))
    assert_that(result).is_instance_of(MergeResult)
    assert_that(result.baked).is_false()
    assert_that(result.adapter_name).is_equal_to("merged")
    assert_that(result.output_dir).is_equal_to(out)


def test_merge_adapters_bake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bake=True merges into base weights and saves a full checkpoint."""
    a = _adapter_dir(tmp_path, "a_lora")
    b = _adapter_dir(tmp_path, "b_lora")
    out = tmp_path / "baked"
    tok = MagicMock()
    peft_model = MagicMock()
    baked = MagicMock()
    peft_model.merge_and_unload.return_value = baked

    monkeypatch.setattr(
        "slicktune.merge.load_multi_adapters",
        lambda **kwargs: (peft_model, tok),
    )

    result = merge_adapters(
        model_id="fake/base",
        adapters=[
            AdapterRef(path=a, name="a_lora"),
            AdapterRef(path=b, name="b_lora"),
        ],
        output_dir=out,
        method="dare_ties",
        density=0.5,
        bake=True,
    )

    peft_model.merge_and_unload.assert_called_once()
    baked.save_pretrained.assert_called_once_with(str(out))
    tok.save_pretrained.assert_called_once_with(str(out))
    assert_that(result.baked).is_true()
    assert_that(result.adapter_name).is_none()


def test_merge_adapters_unknown_method(tmp_path: Path) -> None:
    """Unknown combination types raise ValueError."""
    a = _adapter_dir(tmp_path, "a")
    with pytest.raises(ValueError, match="Unknown merge method"):
        merge_adapters(
            model_id="fake/base",
            adapters=[AdapterRef(path=a, name="a")],
            output_dir=tmp_path / "out",
            method="not-a-method",
        )


def test_merge_adapters_combined_name_collision(tmp_path: Path) -> None:
    """combined_name must not collide with an input adapter name."""
    a = _adapter_dir(tmp_path, "a")
    b = _adapter_dir(tmp_path, "b")
    with pytest.raises(ValueError, match="collides"):
        merge_adapters(
            model_id="fake/base",
            adapters=[
                AdapterRef(path=a, name="merged"),
                AdapterRef(path=b, name="b"),
            ],
            output_dir=tmp_path / "out",
            combined_name="merged",
        )


def _adapter_dir_with_rank(root: Path, name: str, *, r: int) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        f'{{"base_model_name_or_path": "fake/base", "r": {r}}}',
        encoding="utf-8",
    )
    return path


def test_merge_adapters_mismatched_ranks(tmp_path: Path) -> None:
    """ties/dare/linear reject adapters with different LoRA r before loading."""
    a = _adapter_dir_with_rank(tmp_path, "a", r=16)
    b = _adapter_dir_with_rank(tmp_path, "b", r=8)
    with pytest.raises(ValueError, match="same LoRA r"):
        merge_adapters(
            model_id="fake/base",
            adapters=[
                AdapterRef(path=a, name="a"),
                AdapterRef(path=b, name="b"),
            ],
            output_dir=tmp_path / "out",
            method="ties",
        )


def test_flatten_adapter_save(tmp_path: Path) -> None:
    """Non-default PEFT adapter subdirs are promoted to the checkpoint root."""
    from slicktune.merge import _flatten_adapter_save

    nested = tmp_path / "merged"
    nested.mkdir()
    (nested / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (nested / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    _flatten_adapter_save(output_dir=tmp_path, adapter_name="merged")

    assert_that((tmp_path / "adapter_config.json").is_file()).is_true()
    assert_that((tmp_path / "adapter_model.safetensors").is_file()).is_true()
    assert_that(nested.exists()).is_false()


def test_flatten_adapter_save_noop_when_root_has_config(tmp_path: Path) -> None:
    """Skip flatten when adapter_config.json already exists at the root."""
    from slicktune.merge import _flatten_adapter_save

    nested = tmp_path / "merged"
    nested.mkdir()
    (nested / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (tmp_path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")

    _flatten_adapter_save(output_dir=tmp_path, adapter_name="merged")
    assert_that((tmp_path / "adapter_config.json").read_text(encoding="utf-8")).contains('"r": 16')
    assert_that(nested.exists()).is_true()


def test_flatten_adapter_save_skips_existing_and_tolerates_rmdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip colliding names and ignore OSError when removing the nested dir."""
    from slicktune import merge as merge_mod

    nested = tmp_path / "merged"
    nested.mkdir()
    (nested / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (nested / "leftover.txt").write_text("x", encoding="utf-8")
    (tmp_path / "leftover.txt").write_text("keep", encoding="utf-8")

    real_rmdir = Path.rmdir

    def _rmdir(self: Path) -> None:
        if self.name == "merged":
            raise OSError("not empty")
        real_rmdir(self)

    monkeypatch.setattr(merge_mod.Path, "rmdir", _rmdir)
    merge_mod._flatten_adapter_save(output_dir=tmp_path, adapter_name="merged")

    assert_that((tmp_path / "adapter_config.json").is_file()).is_true()
    assert_that((tmp_path / "leftover.txt").read_text(encoding="utf-8")).is_equal_to("keep")


def test_merge_adapters_matching_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching LoRA r allows ties merge to proceed."""
    a = _adapter_dir_with_rank(tmp_path, "a", r=16)
    b = _adapter_dir_with_rank(tmp_path, "b", r=16)
    tok = MagicMock()
    peft_model = MagicMock()
    monkeypatch.setattr(
        "slicktune.merge.load_multi_adapters",
        lambda **kwargs: (peft_model, tok),
    )

    result = merge_adapters(
        model_id="fake/base",
        adapters=[
            AdapterRef(path=a, name="a"),
            AdapterRef(path=b, name="b"),
        ],
        output_dir=tmp_path / "out",
        method="ties",
        density=0.5,
    )
    peft_model.add_weighted_adapter.assert_called_once()
    assert_that(result.baked).is_false()


def test_bake_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bake_adapter loads one adapter and merge_and_unloads into output_dir."""
    adapter = _adapter_dir(tmp_path, "sft_lora")
    out = tmp_path / "full"
    tok = MagicMock()
    base = MagicMock()
    peft_model = MagicMock()
    baked = MagicMock()
    peft_model.merge_and_unload.return_value = baked
    peft_cls = MagicMock()
    peft_cls.from_pretrained.return_value = peft_model

    monkeypatch.setattr("slicktune.merge.load_tokenizer", lambda model_id: tok)
    monkeypatch.setattr("slicktune.merge._load_base_model", lambda *, model_id: base)
    monkeypatch.setattr("slicktune.merge._maybe_to_device", lambda model: model)
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        SimpleNamespace(PeftModel=peft_cls),
    )
    monkeypatch.setattr("peft.PeftModel", peft_cls, raising=False)

    result = bake_adapter(adapter_dir=adapter, output_dir=out)

    peft_cls.from_pretrained.assert_called_once_with(base, str(adapter))
    peft_model.merge_and_unload.assert_called_once()
    baked.save_pretrained.assert_called_once_with(str(out))
    assert_that(result.baked).is_true()
    assert_that(result.output_dir).is_equal_to(out)


def test_bake_adapter_requires_base_id(tmp_path: Path) -> None:
    """bake_adapter needs model_id when config omits base_model_name_or_path."""
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="model_id is required"):
        bake_adapter(adapter_dir=path, output_dir=tmp_path / "out")


def test_package_exports_merge_api() -> None:
    """Root package re-exports merge helpers."""
    import slicktune

    assert_that(slicktune.AdapterRef).is_equal_to(AdapterRef)
    assert_that(slicktune.MergeResult).is_equal_to(MergeResult)
    assert_that(slicktune.merge_adapters).is_equal_to(merge_adapters)
    assert_that(slicktune.bake_adapter).is_equal_to(bake_adapter)
    assert_that(slicktune.load_multi_adapters).is_equal_to(load_multi_adapters)


def test_merge_adapters_requires_adapters() -> None:
    """merge_adapters rejects an empty adapter list."""
    with pytest.raises(ValueError, match="At least one adapter"):
        merge_adapters(
            model_id="fake/base",
            adapters=[],
            output_dir="out",
        )


def test_merge_adapters_linear_skips_density(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """linear merges omit density from add_weighted_adapter kwargs."""
    a = _adapter_dir(tmp_path, "a_lora")
    b = _adapter_dir(tmp_path, "b_lora")
    tok = MagicMock()
    peft_model = MagicMock()
    monkeypatch.setattr(
        "slicktune.merge.load_multi_adapters",
        lambda **kwargs: (peft_model, tok),
    )

    merge_adapters(
        model_id="fake/base",
        adapters=[
            AdapterRef(path=a, name="a_lora"),
            AdapterRef(path=b, name="b_lora"),
        ],
        output_dir=tmp_path / "out",
        method="linear",
        density=0.5,
    )

    kwargs = peft_model.add_weighted_adapter.call_args.kwargs
    assert_that(kwargs).does_not_contain_key("density")
    assert_that(kwargs["combination_type"]).is_equal_to("linear")


def test_merge_adapters_save_without_selected_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to plain save_pretrained when selected_adapters is unsupported."""
    a = _adapter_dir(tmp_path, "a_lora")
    tok = MagicMock()
    peft_model = MagicMock()

    def _save(path: str, selected_adapters: list[str] | None = None) -> None:
        if selected_adapters is not None:
            raise TypeError("unexpected keyword")

    peft_model.save_pretrained.side_effect = _save
    monkeypatch.setattr(
        "slicktune.merge.load_multi_adapters",
        lambda **kwargs: (peft_model, tok),
    )

    merge_adapters(
        model_id="fake/base",
        adapters=[AdapterRef(path=a, name="a_lora")],
        output_dir=tmp_path / "out",
        method="linear",
        density=None,
    )
    assert_that(peft_model.save_pretrained.call_count).is_equal_to(2)


def test_load_multi_adapters_not_a_directory(tmp_path: Path) -> None:
    """Non-directory adapter paths raise ValueError."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        load_multi_adapters(
            model_id="fake/base",
            adapters=[AdapterRef(path=file_path, name="a")],
        )


def test_default_adapter_name_sanitizes() -> None:
    """Special characters in path stems are replaced."""
    from slicktune.merge import _default_adapter_name

    assert_that(_default_adapter_name(path=Path("foo/bar.baz!"))).is_equal_to("bar_baz_")


def test_load_base_model_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA loads use device_map=auto and skip _maybe_to_device."""
    from slicktune.merge import _load_base_model

    model = MagicMock()
    captured: dict[str, Any] = {}

    def _from_pretrained(model_id: str, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        return model

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("slicktune.merge.resolve_dtype", lambda: "float32")
    monkeypatch.setattr(
        "slicktune.merge.AutoModelForCausalLM.from_pretrained",
        _from_pretrained,
    )
    monkeypatch.setattr(
        "slicktune.merge._maybe_to_device",
        lambda m: (_ for _ in ()).throw(AssertionError("should not move")),
    )

    loaded = _load_base_model(model_id="fake/base")
    assert_that(loaded).is_equal_to(model)
    assert_that(captured["kwargs"]).contains_key("device_map")


def test_load_base_model_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-CUDA loads move via _maybe_to_device."""
    from slicktune.merge import _load_base_model

    model = MagicMock()
    moved = MagicMock()

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("slicktune.merge.resolve_dtype", lambda: "float32")
    monkeypatch.setattr(
        "slicktune.merge.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: model,
    )
    monkeypatch.setattr("slicktune.merge._maybe_to_device", lambda m: moved)

    assert_that(_load_base_model(model_id="fake/base")).is_equal_to(moved)


def test_maybe_to_device_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device helper respects device_map, MPS, CUDA, then CPU."""
    from slicktune.merge import _maybe_to_device

    mapped = MagicMock()
    mapped.hf_device_map = {"layer": 0}
    assert_that(_maybe_to_device(mapped)).is_equal_to(mapped)

    mps_model = MagicMock()
    mps_model.hf_device_map = None
    mps_model.to.return_value = mps_model
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert_that(_maybe_to_device(mps_model)).is_equal_to(mps_model)
    mps_model.to.assert_called_with("mps")

    cuda_model = MagicMock()
    cuda_model.hf_device_map = None
    cuda_model.to.return_value = cuda_model
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert_that(_maybe_to_device(cuda_model)).is_equal_to(cuda_model)
    cuda_model.to.assert_called_with("cuda")

    cpu_model = MagicMock()
    cpu_model.hf_device_map = None
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert_that(_maybe_to_device(cpu_model)).is_equal_to(cpu_model)
