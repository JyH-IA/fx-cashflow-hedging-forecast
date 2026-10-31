# Open questions and items to verify

Last reviewed on 2026-10-31. Items marked `TODO(verify)` in the code and configuration are listed here, plus known gaps.

## Verified

- CI passes on the pull request that redesigned the dashboard: lint, `mypy --strict`, tests (coverage gate of 80 %), secret scan, dependency audit and the Docker build.
- Real LightGBM and Optuna tuning, the FastAPI app, the Streamlit smoke test and the SQLite tests run in CI.
- `make demo` ran on a developer machine.
- Ingestion with a real Banxico token works: the URL template and the shape of the JSON response are correct, and the Yahoo Finance fallback is available. It ran on 2026-10-31.
- Series IDs: `SF43718` (FIX) and `SF60633` (28-day CETES) were checked against Banxico's documentation, and `DFF` against FRED. `SF43783`, used before for `cetes_28d`, is the 28-day TIIE and was replaced.

## Series and data still to check

- `SP1` (INPC) in `configs/series.yaml` is not checked against Banxico's documentation.
- SECTUR has no API assumed here: arrivals are a manual CSV download. The file was not provided in the real-data run, so `cancun_arrivals` is left out.
- The interest rates in `configs/business.yaml` (6.22 % and 3.88 %) are illustrative inputs for the forward and must be updated by hand.
- `make ingest` stops at the end date of `configs/simulator.yaml` and fails if `DB_URL` is set and the database is down. Making the database step optional, or failing softly, is not done.
- The internal data (sales, occupancy, cash-in) is synthetic, so the cash-in results do not describe a real business.

## Gate, base model and backtest

- In the real-data run (2026-10-31) the gate rejected the residual tree for both targets in the stored models: `y_fx` repeats the last observed value and `y_cash` is Prophet alone. The dashboard's Technical details tab shows this. The backtest agrees for `y_fx`: its hybrid equals the random walk (ratio 1.000 with identical errors).
- `reports/shap_drivers.md` can be stale: it listed drivers of a `y_fx` residual model after training had rejected that tree. Run `make explain` after every `make train`; it should report that there is nothing to explain for both targets.
- On the synthetic series the log showed a gate rejection for `y_fx` in 3 of 5 folds, so the tree was kept in the other 2 and the hybrid ended above the random walk. Only rejections are logged, so the validation errors of the kept trees are unknown; logging every gate decision is not done.
- The backtest has five origins (`n_splits: 5`) and no confidence intervals or significance test. Gaps of a few percent against the random walk, such as the seasonal naive (7 days) at 0.917 to 0.952, cannot be told from chance. More origins and the error of each origin would allow an interval; this is not done.
- The gate uses one validation window of about 270 rows (three blocks of 90 days) and `gate_min_improvement` is 0, so a small, noisy gain is enough to keep a tree. A margin such as 0.10 has not been tested.
- `BASE_BY_TARGET` is fixed in code. Making it configurable, or choosing the base by backtest, is not done.
- When the tree is rejected, the forecast ignores the scenario shocks, because only the tree uses the features. The dashboard scenarios therefore show the same point forecast for both targets.
- Prophet warns that yearly seasonality is under-identified with less than 730 days of history, which applies to the first backtest folds.

## Not exercised by tests or not run end to end

- Real Prophet fitting and its JSON serialization (tests use a lightweight stand-in; it runs locally).
- Real XGBoost training and JSON artifacts, and saving and loading real LightGBM artifacts.
- The validation gate with a real LightGBM tree (tests use stub trees).
- SHAP `TreeExplainer` on a loaded booster (it runs locally with `make explain` when a tree is kept).
- `make api`, Docker Compose, drift monitoring (`make monitor`) and the PostgreSQL and Oracle schemas.

## Tooling

- Docker base images are not pinned by digest.
- A container image scan (for example Trivy) is not configured; add it once the action version can be verified.
- The dependency audit stays advisory because new advisories can appear at any time.
- `starlette.testclient` warns that `httpx` is deprecated in favour of a newer package; revisit when upgrading FastAPI.

## Not implemented

- MLflow tracking and a model registry.
- A hashed `requirements.lock` (generate with `pip-compile --generate-hashes` when ready).
- A Java/Spring Boot client example. The OpenAPI contract can be exported with `make openapi`, but `docs/openapi.json` is not versioned.
- Walk-forward model forecasts as input to the forecast-guided hedging strategy.
- Prediction intervals calibrated on backtest errors, and a real VaR.

## Product questions

- Should hedge ratios be capped by policy, and what are the real bank spreads and minimum tickets?
- Is the monthly decision cadence the right one for the treasury team?
