# Open questions and items to verify

Items marked `TODO(verify)` in the code and configuration, plus known gaps.

## Status after the first CI run (2026-10-29)

- 158 of 160 tests passed with 84.8 % coverage. The two failures were a real bug: `:name` tokens inside SQL comments were treated as bind parameters. The comments were fixed and `tests/test_sql_files.py` guards against it.
- Real LightGBM and Optuna tuning, the FastAPI app, the Streamlit smoke test and the SQLite tests ran in CI.
- `ruff` found three style issues (a long line, a hardcoded-password false positive in a test and a nested `with`), now fixed.
- The Trivy action version was invalid and the step was removed; the Docker build job now only builds the image.
- `pip-audit` reported vulnerabilities in the `setuptools` build tool of the runner, not in project dependencies; CI now upgrades it.

## External identifiers and APIs

- Banxico SIE series IDs in `configs/series.yaml` (`SF43718`, `SF43783`, `SP1`) and the FRED series `DFF`: confirm against the providers' documentation.
- Shape of the Banxico JSON response and the URL template used by `data_ingestion`: confirm with a real token.
- SECTUR has no API assumed here: arrivals are a manual CSV download.

## Still not exercised by tests

- Real Prophet fitting and its JSON serialization (tests use a lightweight stand-in).
- Real XGBoost training and JSON artifacts, and saving/loading real LightGBM artifacts.
- SHAP `TreeExplainer` on a loaded booster.
- The Docker image build (the job now runs for the first time).

## Tooling

- Docker base images are not pinned by digest.
- `mypy --strict` reports about 30 errors in five modules; it stays advisory until they are fixed.
- A container image scan (for example Trivy) is not configured; add it once the action version can be verified.
- The coverage gate is 80 %.
- `starlette.testclient` warns that `httpx` is deprecated in favour of a newer package; revisit when upgrading FastAPI.

## Not implemented

- MLflow tracking and a model registry.
- A hashed `requirements.lock` (generate with `pip-compile --generate-hashes` when ready).
- A Java/Spring Boot client example and the exported `docs/openapi.json` (run `make openapi`).
- Walk-forward model forecasts as input to the forecast-guided hedging strategy.
- Prediction intervals calibrated on backtest errors, and a real VaR.

## Product questions

- Should hedge ratios be capped by policy, and what are the real bank spreads and minimum tickets?
- Is the monthly decision cadence the right one for the treasury team?
