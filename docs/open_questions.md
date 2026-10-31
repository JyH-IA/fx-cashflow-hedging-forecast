# Open questions and items to verify

Last reviewed on 2026-10-31. Items marked `TODO(verify)` in the code and configuration are listed here, plus known gaps.

## Verified

- CI passes on the pull request that redesigned the dashboard: lint, `mypy --strict`, tests (coverage gate of 80 %), secret scan, dependency audit and the Docker build.
- Real LightGBM and Optuna tuning, the FastAPI app, the Streamlit smoke test and the SQLite tests run in CI.
- `make demo` ran on a developer machine.
- Ingestion with a real Banxico token works: the URL template and the shape of the JSON response are correct, and the Yahoo Finance fallback is available. It ran on 2026-10-31.
- Series IDs: `SF43718` (FIX) and `SF60633` (28-day CETES) were checked against Banxico's documentation, and `DFF` against FRED. `SF43783`, used before for `cetes_28d`, is the 28-day TIIE and was replaced.
- `SP1` (INPC) was checked against the published index: the downloaded data holds 145.462 for August 2026 and 146.075 for September 2026, the published values.
- After the real-data training of 2026-10-31, `make explain` reported that the gate disabled the residual model for both targets, so there is nothing to explain with SHAP. An older `reports/shap_drivers.md` had listed drivers of a `y_fx` residual model from a previous run; run `make explain` after every `make train` to keep the file current.
- The sales simulator draws the noise of each component from its own random stream, so extending the end date keeps the earlier days (`test_extending_the_end_date_keeps_the_earlier_days`). Before 2026-10-31, moving the end date re-drew the sales and cash-in noise of the whole history.

## Series and data still to check

- The figures of the README and of the executive summary were produced before the simulator fix and with an end date set by hand. Run `make ingest features train evaluate-full hedge explain` again and refresh them.
- SECTUR has no API assumed here: arrivals are a manual CSV download. The file was not provided in the real-data run, so `cancun_arrivals` is left out.
- The interest rates in `configs/business.yaml` (6.22 % and 3.88 %) are illustrative inputs for the forward and must be updated by hand.
- `make ingest` ends yesterday unless `--end` is given, and it fails if `DB_URL` is set and the database is down. Making the database step optional, or failing softly, is not done.
- The synthetic macro generator of `make sample` draws its series one after the other from a single random stream: the exchange-rate walk keeps its earlier days when the dates change, but the Yahoo Finance noise and the CETES series do not.
- The internal data (sales, occupancy, cash-in) is synthetic, so the cash-in results do not describe a real business.

## Gate, base model and backtest

- In the real-data run (2026-10-31) the gate rejected the residual tree for both targets in the stored models: `y_fx` repeats the last observed value and `y_cash` is Prophet alone. The dashboard's Technical details tab shows this. The backtest agrees for `y_fx`: its hybrid equals the random walk (ratio 1.000 with identical errors).
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
- `scripts/diagnose_hybrid.py` has no tests.
- `make api`, Docker Compose, drift monitoring (`make monitor`) and the PostgreSQL and Oracle schemas.

## Tooling

- CI runs `typecheck` and `test` on Python 3.11 and 3.12 since 2026-10-31. The first results of the 3.12 jobs are still to be checked.
- The drift report workflow (`monitoring.yml`) is manual and uses synthetic data.
- The Docker `app` service does not mount `reports/`, so the reliability tab has no data in the container.
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
