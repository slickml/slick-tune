🛠 Installation
=================

- Install a [Python version >=3.10,<3.13](https://www.python.org).
- All SlickTune development uses [*uv*](https://docs.astral.sh/uv/). Install `uv` from the
  [*installation docs*](https://docs.astral.sh/uv/getting-started/installation/), then sync 🏃‍♀️:

  ```bash
  uv sync
  ```

- QLoRA (CUDA + bitsandbytes only) 🔥:

  ```bash
  uv sync --extra qlora
  ```

- Documentation build extras (Sphinx / Furo / MyST):

  ```bash
  uv sync --group docs
  ```

- From PyPI (when published):

  ```bash
  pip install slicktune
  # or
  uv add slicktune
  ```

- CLI entry point:

  ```bash
  uv run slicktune --help
  ```

- Task runner is [Poe the Poet](https://poethepoet.natn.io/installation.html):

  ```bash
  uv tool install poethepoet
  poe greet
  poe sphinx
  ```

- Prefer an isolated environment (`uv`, `venv`, or `conda`) so training deps do not collide
  with other projects.
