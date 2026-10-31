# Contributing

Thanks for your interest. This is a portfolio project, but fixes and suggestions are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make setup          # installs the package with dev tools and the pre-commit hooks
make demo           # synthetic data -> features -> models -> evaluation (no token needed)
```

## Workflow

1. Create a branch from `main`.
2. Keep changes small and add tests for new behaviour.
3. Run `make format lint test` before opening a pull request.
4. Use Conventional Commits for messages (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).

## Code style

- Python 3.11 or 3.12 (CI runs 3.11), full type annotations, and docstrings with the raised errors.
- Pure functions where possible; keep network, disk and database access at the edges.
- Configuration lives in `configs/*.yaml` and is validated with pydantic.

## Data and secrets

- Use synthetic data in tests and examples. Never commit real company data.
- Never commit tokens, `.env` files or credentials. See `SECURITY.md`.
