# Security Policy

This is an educational portfolio project. All internal business data is synthetic.

## Secrets policy

- Secrets (for example `BANXICO_TOKEN`, `DB_URL`) are read only from environment variables.
- Never commit `.env`. Only `.env.example` with placeholder values is versioned.
- Never pass secrets with Docker `ENV` or `ARG`; use `--env-file` or runtime secrets.
- Notebooks are stripped of outputs with `nbstripout`; `gitleaks` runs in pre-commit and CI.

## If a secret leaks

1. Revoke the token at the provider (for Banxico, request a new SIE token) and generate a new one.
2. Remove the secret from the codebase and rotate any related credentials.
3. Rewrite the git history only as an additional measure; revocation is mandatory.

## Model artifacts

Models are serialized as JSON and verified with a SHA-256 hash before loading. Pickle files are not allowed.

## Reporting a vulnerability

Please open a private security advisory in this repository (Security tab). Do not include secrets in public issues.

## Disclaimer

This project is not financial advice. Forecasts and hedging simulations are illustrative.
