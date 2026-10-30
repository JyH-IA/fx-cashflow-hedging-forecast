# Open questions and items to verify

Items marked `TODO(verify)` in the code and configuration, plus known gaps.

## Status after the first CI runs (2026-10-29)

- All six CI jobs pass: lint, `mypy --strict`, tests (160+ tests, about 85 % coverage), secret scan, dependency audit and the Docker build.
- Real LightGBM and Optuna tuning, the FastAPI app, the Streamlit smoke test and the SQLite tests ran in CI.
- The first run found a real bug: `:name` tokens inside SQL comments were treated as bind parameters by SQLAlchemy. It is fixed and `tests/test_sql_files.py` guards against it.
- The Trivy action version used at first did not exist, so the image scan step was removed.
- `mypy --strict` is now a blocking job.

## Status after the validation gate (2026-10-30)

- CI passes with the validation gate, the carry-forward validation and the random-walk base for `y_fx`.
- In a local `make evaluate-full` run, the `y_cash` hybrid (Prophet base, tree rejected) was the best model at 30, 60 and 90 days (`rmse_vs_random_walk` 0.49, 0.42 and 0.38). The `y_fx` hybrid still trailed the random walk (1.56, 1.27 and 1.15) while XGBoost was slightly better than it (0.89 to 0.95). Regenerate the figures with `make evaluate-full`; they depend on the data in use.
- Training with the gate rejected the tree for both targets, so the stored `y_fx` model is the random walk and the stored `y_cash` model is Prophet alone.

## Gate and base model

- In the last `y_fx` backtest (random-walk base) the log shows a gate rejection in 3 of 5 folds, so the tree was kept in the other 2. The hybrid ends above the random walk (1.15 to 1.56), while a random-walk base with a rejected tree would equal it exactly, so the two kept trees most likely hurt. The count of five folds comes from `n_splits: 5` and from the five rejections logged for `y_cash` in the same run. Only rejections are logged, so the validation errors of the kept trees are unknown; logging every gate decision is not done.
- The `y_cash` gate rejected the tree in all five folds, so its backtest hybrid is Prophet alone.
- The gate uses one validation window of about 270 rows (three blocks of 90 days) and `gate_min_improvement` is 0, so a small, noisy gain is enough to keep a tree. A margin such as 0.10 has not been tested, and it is not known whether it would have rejected the two kept `y_fx` trees.
- `BASE_BY_TARGET` is fixed in code. Making it configurable, or choosing the base by backtest, is not done.
- When the tree is rejected, the forecast ignores the scenario shocks, because only the tree uses the features. The dashboard scenarios for that target therefore show the same point forecast.
- Prophet warns that yearly seasonality is under-identified with less than 730 days of history, which applies to the first backtest folds.

## External identifiers and APIs

- Banxico SIE series IDs in `configs/series.yaml` (`SF43718`, `SF43783`, `SP1`) and the FRED series `DFF`: confirm against the providers' documentation.
- Shape of the Banxico JSON response and the URL template used by `data_ingestion`: confirm with a real token.
- SECTUR has no API assumed here: arrivals are a manual CSV download.

## Still not exercised by tests

- Real Prophet fitting and its JSON serialization (tests use a lightweight stand-in).
- Real XGBoost training and JSON artifacts, and saving/loading real LightGBM artifacts.
- The validation gate with a real LightGBM tree (tests use stub trees).
- SHAP `TreeExplainer` on a loaded booster.
- The full `make demo` pipeline on a developer machine.

## Tooling

- Docker base images are not pinned by digest.
- A container image scan (for example Trivy) is not configured; add it once the action version can be verified.
- The dependency audit stays advisory because new advisories can appear at any time.
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
