📣 🥁 Changelog & Releases
===========================

- We follow [Semantic Versioning](http://semver.org/) to document any notable changes.
- Please checkout [SlickTune Official Releases](https://github.com/slickml/slick-tune/releases) for more details.


## 📍 Unreleased Version X.X.X - XXXX-XX-XX
### 🛠 Fixed

### 🔥 Added

---

## 📍 Version 0.1.0 - 2026-07-14
### 🛠 Fixed
- Fixed AdaLoRA training callback to call `update_and_allocate` on `on_optimizer_step` (grads must exist before `zero_grad`).
- Fixed AdaLoRA tiny-SFT defaults (warmup `tinit` / `tfinal`, schedule estimation) so probes can pass.
- Fixed LLM judge score parsing (ignore `0-10` scale echoes) and digit-constrained scoring for small judge models.

### 🔥 Added
- Initial Phase 1–2 library: composable `Tuner` with `model × strategy × objective × data × metrics`.
- Strategies: `LoRAStrategy`, `DoRAStrategy`, `AdaLoRAStrategy`, `QLoRAStrategy`, `FullStrategy`.
- Objective: `SFTObjective` (DPO stub for Phase 3).
- Data loaders for SFT / probe JSONL and shipped `about_amir` train / eval / probe datasets.
- Metrics tracker, probe recipes, holdout perplexity, `SubstringJudge` / `LLMJudge`.
- CLI: `slick-tune train` / `probe` / `eval` plus example scripts and Poe tasks.
- Sphinx docs (Furo + AutoAPI + MyST) under `docs/`, including the Fine-Tuning Guide.
- CI (uv, Ubuntu + macOS, Python 3.10–3.12), Codecov, 100% unit-test coverage gate.

---
