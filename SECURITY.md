# Security Policy

This is an educational portfolio project. All internal business data is synthetic and every figure is illustrative. Nothing here is financial advice.

## Secrets policy

- Secrets (`BANXICO_TOKEN`, `DB_URL`, database passwords) are read only from environment variables, through `pydantic-settings` and `SecretStr`, so they never appear in `repr`, logs or error messages.
- Never commit `.env`. Only `.env.example` with placeholder values is versioned.
- The Banxico token is sent in the `Bmx-Token` header, never in the URL.
- Never pass secrets to Docker with `ENV` or `ARG`; use `--env-file` or runtime secrets.
- Do not paste real tokens or company data into prompts for AI tools, issues or pull requests.
- Protections in place: `gitleaks` and `detect-private-key` in pre-commit and in CI, and `nbstripout` for notebooks. Also enable GitHub secret scanning and push protection in the repository settings.

## If a secret leaks

1. Revoke the token at the provider (for Banxico, request a new SIE token) and generate a new one.
2. Remove the secret from the codebase and rotate any related credentials.
3. Rewrite the git history only as an additional measure; revocation is mandatory.

## Model artifacts

Models are never stored as pickle files. Prophet and XGBoost are saved as JSON and LightGBM in its text format; all of them are data formats that do not execute code. `metadata.json` records a SHA-256 hash of every artifact, and models are verified against it before loading. Only load artifacts that you trained yourself or that come from a source you trust.

## Deployment notes

- The Streamlit app and the API bind to `127.0.0.1` by default. Do not expose them publicly without authentication or a reverse proxy with TLS.
- The Docker image runs as a non-root user and contains no models, data or secrets; mount them at runtime.
- The base image is not yet pinned by digest (`TODO(verify)` in the `Dockerfile`).

## Dependencies

Dependabot proposes updates weekly, and `pip-audit` runs in CI as an advisory check.

## Reporting a vulnerability

Please open a private security advisory in this repository (Security tab). Do not include secrets in public issues.
