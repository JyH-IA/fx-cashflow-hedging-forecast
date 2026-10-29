# Open questions and items to verify

Items marked `TODO(verify)` in the code and configuration, plus known gaps.

## External identifiers and APIs

- Banxico SIE series IDs in `configs/series.yaml` (`SF43718`, `SF43783`, `SP1`) and the FRED series `DFF`: confirm against the providers' documentation.
- Shape of the Banxico JSON response and the URL template used by `data_ingestion`: confirm with a real token.
- SECTUR has no API assumed here: arrivals are a manual CSV download.

## Libraries exercised only in CI

- Prophet fitting and JSON serialization, LightGBM text artifacts, XGBoost JSON artifacts, Optuna tuning and SHAP `TreeExplainer` on loaded boosters.
- The FastAPI application, the Streamlit smoke test and the SQLAlchemy/SQLite tests.

## Tooling

- Docker base images are not pinned by digest.
- The Trivy action version in `ci.yml` should be confirmed.
- `mypy --strict`, `pip-audit` and the image scan are advisory until the first full CI run is clean.
- The coverage gate is set to 70 %; raise it once the real figure is known.

## Not implemented

- MLflow tracking and a model registry.
- A hashed `requirements.lock` (generate with `pip-compile --generate-hashes` when ready).
- A Java/Spring Boot client example and the exported `docs/openapi.json` (run `make openapi`).
- Walk-forward model forecasts as input to the forecast-guided hedging strategy.
- Prediction intervals calibrated on backtest errors, and a real VaR.

## Product questions

- Should hedge ratios be capped by policy, and what are the real bank spreads and minimum tickets?
- Is the monthly decision cadence the right one for the treasury team?
