# Contributing

Thanks for your interest in improving lora-tools! This is a small hobby toolkit,
but PRs and issues are welcome.

## Development setup

Everything runs through [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Brioch/lora-tools
cd lora-tools
uv sync --group dev          # create the dev environment
uvx pre-commit install       # install the git hook
```

## Running the checks

CI runs exactly these — run them locally before opening a PR:

```bash
uv run --group dev pytest --cov=tools   # tests + coverage gate (>= 90%)
uv run --group dev mypy                 # strict type checking
uvx pre-commit run --all-files          # ruff lint + format + mypy + hygiene
```

## Conventions

- **Style & imports:** [Ruff](https://docs.astral.sh/ruff/) (lint + format). The
  pre-commit hook auto-fixes most issues.
- **Types:** all code in `tools/` is fully annotated and must pass `mypy --strict`.
- **Tests:** add tests under `tests/` for any behaviour change; keep coverage at
  or above the gate. Pure logic is tested directly; CLIs are driven in-process by
  patching `sys.argv`.
- **Scripts are standalone:** each `tools/*.py` declares its own dependencies via
  a [PEP 723](https://peps.python.org/pep-0723/) header and must remain runnable
  with `uv run tools/<script>.py`.
- **Docs:** user-facing changes go in [`docs/`](docs/) (usage in
  [`docs/tools.md`](docs/tools.md)); keep the README a summary.

## Commit messages

Conventional-commit style is appreciated (`feat:`, `fix:`, `docs:`, `chore:`,
`test:`) and keeps the changelog easy to assemble.
