📣 🥁 Changelog & Releases
===========================

- We follow [Semantic Versioning](http://semver.org/) to document any notable changes.
- Please checkout [SlickTune Official Releases](https://github.com/slickml/slick-tune/releases) for more details.


## 📍 Unreleased Version X.X.X - XXXX-XX-XX
### 🛠 Fixed
- Moved ruff config from `pyproject.toml` to `ruff.toml`; moved pytest options to `pytest.ini`.
- Pointed `CONTRIBUTING.md` at `ruff.toml` / `pytest.ini`.
- Ran `poe sphinx` in the tox env (docs build gate).

### 🔥 Added
- `GRPOObjective` with TRL `GRPOTrainer` and verifiable `substring_must_contain_reward`.
- Soft keyword-overlap fallback in the GRPO reward so groups can get non-zero advantages.
- `Tuner.adapter_path` to warm-start PEFT adapters with `is_trainable=True` (SFT → GRPO).
- `load_grpo_jsonl` + demo `about_amir.grpo.jsonl` (prompt + `must_contain` / `solution`).
- CLI `--objective grpo`, `--num-generations`, `--max-completion-length`; example `run_grpo_lora.py` + `poe train-grpo`.

---

## 📍 Version 0.3.0 - 2026-07-15
### 🛠 Fixed
- Auto-bump KTO `per_device_train_batch_size` to at least 2 (TRL requires batch size > 1 for the KL term).
- Moved mypy config from `pyproject.toml` to `mypy.ini`; added `tox.ini` (`poe tox`).
- Fixed tox to use `tox-uv` (`uv-venv-lock-runner` + `only-managed`) so each env uses uv-managed CPython from the env name (not the project `.venv`).

### 🔥 Added
- Preference objectives: `DPOObjective`, `ORPOObjective` (TRL experimental), `KTOObjective`.
- Preference / KTO JSONL loaders and demo datasets (`about_amir.prefs.jsonl`, `about_amir.kto.jsonl`).
- CLI `--objective sft|dpo|orpo|kto` and `--beta`; examples `run_dpo_lora.py` / `run_kto_lora.py` + `poe train-dpo` / `poe train-kto`.
- `tox` for multi-Python check/test/build (aligned with afk-bot).
- Python 3.13 support (`requires-python = ">=3.10,<3.14"`, CI + tox matrix).

---

## 📍 Version 0.2.0 - 2026-07-14
### 🛠 Fixed
- Fixed AdaLoRA training callback to call `update_and_allocate` on `on_optimizer_step` (grads must exist before `zero_grad`).
- Fixed AdaLoRA tiny-SFT defaults (warmup `tinit` / `tfinal`, schedule estimation) so probes can pass.
- Fixed LLM judge score parsing (ignore `0-10` scale echoes) and digit-constrained scoring for small judge models.
- Fixed AdaLoRA schedule `dataclasses.replace` typing for mypy.
- Renamed CLI entry point from `slick-tune` to `slicktune` (matches the package name).
- Hardened docs CD: deploy `_static/` in a separate FTP sync so the host does not FIN mid-session on first publish.

### 🔥 Added
- Strategies: `DoRAStrategy` and `AdaLoRAStrategy` (+ `AdaLoRACallback`).
- Holdout eval stack: `slicktune eval`, perplexity, `SubstringJudge` / `LLMJudge`, and `about_amir.eval.jsonl`.
- Example scripts for DoRA / AdaLoRA SFT runs.
- Sphinx docs (Furo + AutoAPI + MyST), Fine-Tuning Visual Guide, and SemVer-style changelog.
- Docs CD workflow: build Sphinx on push to `master` and deploy to `docs.slickml.com/slick-tune/` via FTP.
- Keyword-only public APIs and named-arguments Cursor rule.
- Added `slicktune --version` (Click `version_option`).

---

## 📍 Version 0.1.0 - 2026-07-14
### 🔥 Added
- Initial Phase 1 library: composable `Tuner` with `model × strategy × objective × data × metrics`.
- Strategies: `LoRAStrategy`, `QLoRAStrategy`, `FullStrategy`.
- Objective: `SFTObjective` (DPO stub for Phase 3).
- Data loaders for SFT / probe JSONL and shipped `about_amir` train / probe datasets.
- Metrics tracker, probe recipes, and `slicktune train` / `probe` CLI plus example scripts.
- CI (uv, Ubuntu + macOS, Python 3.10–3.12), Codecov, 100% unit-test coverage gate.

---
