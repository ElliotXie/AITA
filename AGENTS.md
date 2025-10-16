# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `aita/` with the Typer CLI entrypoint in `main.py` and runtime configuration in `config.py`. Domain schemas sit under `aita/domain`, service adapters under `aita/services`, orchestration pipelines in `aita/pipelines`, report builders in `aita/report`, and shared helpers in `aita/utils`; Jinja templates live in `aita/templates`. Keep datasets in `data/`, supporting notes in `docs/`, worked examples in `examples/`, and automation utilities in `scripts/`. Tests target fast feedback in `tests/`, while broader workflow regressions live in `test/`; prefer isolating generated artifacts in `tests/diagnostic_output`.

## Build, Test, and Development Commands
`pip install -e .[dev]` provisions the editable package plus linting hooks. `aita --help` surfaces available CLI flows; exercise a path with `aita ingest data/sample_run` before larger changes. `pytest` honors verbose coverage flags from `pyproject.toml`; subset runs with `pytest tests/services/test_ocr_simple.py`. Run `pre-commit run --all-files` to execute Black, isort, Flake8, and mypy locally ahead of review.

## Coding Style & Naming Conventions
Code targets Python 3.8+ with 4-space indents and the 88-character limit enforced by Black. isort’s Black profile keeps imports grouped; fall back to `black . && isort .` when needed. mypy’s `disallow_untyped_defs` means new functions must ship with type hints. Use `CamelCase` for domain models, `snake_case` for functions and variables, and CLI command names that match their dashed Typer verbs (e.g., `generate-rubric`).

## Testing Guidelines
Pytest drives coverage via `--cov=aita --cov-report=term-missing`; maintain clean reports before merging. Name suites `test_*.py` and test functions `test_*` to align with discovery rules. Place scenario fixtures in `tests/pipelines` or `tests/services` rather than `data/` to avoid leaking real submissions. Pair new pipelines with a targeted unit test plus, when applicable, an integration case in `test/` mirroring production exam bundles.

## Commit & Pull Request Guidelines
With no existing history, adopt Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) written in imperative present tense. Squash or rebase away noisy WIP commits before pushing. Each PR should explain user impact, reference related issues, and list verification commands (`pytest`, `pre-commit`). Include screenshots or report paths whenever UI or generated artifacts change, and confirm `.env` secrets stay out of diffs.
