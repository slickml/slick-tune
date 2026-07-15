📣 🥁 Changelog & Releases
===========================

- We follow [Semantic Versioning](http://semver.org/) to document any notable changes.
- Please checkout [SlickTune Official Releases](https://github.com/slickml/slick-tune/releases) for more details.


## 📍 Unreleased Version X.X.X - XXXX-XX-XX
### 🛠 Fixed

### 🔥 Added

---

## 📍 Version 0.2.0 - 2026-07-14
### 🛠 Fixed
- Fixed AdaLoRA training callback to call `update_and_allocate` on `on_optimizer_step` (grads must exist before `zero_grad`).
- Fixed AdaLoRA tiny-SFT defaults (warmup `tinit` / `tfinal`, schedule estimation) so probes can pass.
- Fixed LLM judge score parsing (ignore `0-10` scale echoes) and digit-constrained scoring for small judge models.
- Fixed AdaLoRA schedule `dataclasses.replace` typing for mypy.

### 🔥 Added
- Strategies: `DoRAStrategy` and `AdaLoRAStrategy` (+ `AdaLoRACallback`).
- Holdout eval stack: `slick-tune eval`, perplexity, `SubstringJudge` / `LLMJudge`, and `about_amir.eval.jsonl`.
- Example scripts for DoRA / AdaLoRA SFT runs.
- Sphinx docs (Furo + AutoAPI + MyST), Fine-Tuning Visual Guide, and SemVer-style changelog.
- Docs CD workflow: build Sphinx on push to `master` and deploy to `docs.slickml.com/slick-tune/` via FTP.
- Keyword-only public APIs and named-arguments Cursor rule.

---

## 📍 Version 0.1.0 - 2026-07-14
### 🔥 Added
- Initial Phase 1 library: composable `Tuner` with `model × strategy × objective × data × metrics`.
- Strategies: `LoRAStrategy`, `QLoRAStrategy`, `FullStrategy`.
- Objective: `SFTObjective` (DPO stub for Phase 3).
- Data loaders for SFT / probe JSONL and shipped `about_amir` train / probe datasets.
- Metrics tracker, probe recipes, and `slick-tune train` / `probe` CLI plus example scripts.
- CI (uv, Ubuntu + macOS, Python 3.10–3.12), Codecov, 100% unit-test coverage gate.

---
