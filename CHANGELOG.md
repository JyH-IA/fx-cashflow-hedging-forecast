# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Data contracts, YAML configuration and a SQL layer (PostgreSQL by default, Oracle optional).
- Ingestion from Banxico and Yahoo Finance, plus a seasonal synthetic sales simulator.
- Fully synthetic sample data so the pipeline runs without any token.
- Feature engineering without target leakage (lags and rolling windows on shifted series).
- Prophet + LightGBM hybrid, XGBoost comparator, SARIMA and naive baselines, rolling-origin backtest.
- SHAP explanations, historical hedging simulation with block bootstrap, and prediction intervals.
- Drift monitoring (PSI), a retraining trigger, a FastAPI service and a Streamlit dashboard.
- Docker image, Docker Compose services, CI, Dependabot and security scanning.
- Validation gate for the hybrid's residual tree (`gate_holdout_fraction`, `gate_min_improvement`). The tree is kept only if it lowers the error on the most recent residuals; otherwise the hybrid is its base alone. `--no-gate` turns it off in `make train`.
- The gate scores the tree with exogenous features frozen at the forecast origin, as at inference time (`frozen_holdout_features`).
- Random-walk base (`LastValueBase`) with its own JSON artifact. `metadata.json` now records `base_model`, `residual_model` and the gate result.
- `evaluate-full` runs the gate on every backtest fold, and `explain` skips SHAP for a target whose tree was rejected.
- The dashboard shows a notice for each target whose forecast comes from the base model alone because the gate rejected its tree (`base_only_notices`).
- The dashboard has a model comparison that reads `reports/metrics.json`, written by `make evaluate` and `make evaluate-full`, and shows each model's RMSE relative to the random walk by target and horizon, with one plain sentence per target (`reliability_verdicts`).
- A marker file `data/raw/data_source.txt` (`synthetic` or `banxico`), written by `make sample` and `make ingest` and copied to `data/processed/` by `make features`. The dashboard shows a line under the title with the source, the date of the last observation and the last USD/MXN rate; synthetic or unrecorded data shows as a warning.

### Changed

- The hybrid for `y_fx` uses a random walk as its base instead of Prophet, which lost to a random walk at every origin of the diagnostic. `y_cash` keeps Prophet. The base is chosen per target in `BASE_BY_TARGET`.
- Residuals of the random-walk base are computed with no gap, because that base is refitted up to each forecast origin.
- The dashboard is organized in four tabs (Summary, Hedging, How reliable is it?, Technical details). The Summary opens with a headline and four key figures about what the selected hedge protects, the filters are the planning horizon in months, a named hedge level and the USD amount (shown with a dollar sign and in compact form), and the scenario selector and run date moved to Advanced settings.
- The illustrative rates in `configs/business.yaml` are now 6.22 % for Mexico and 3.88 % for the US.
- The README and `docs/` were trimmed to what was run: data sources, what was checked in CI, run locally and not run, and the real-data steps.

### Fixed

- `configs/series.yaml` mapped `cetes_28d` to `SF43783`, which is the 28-day TIIE. It now uses `SF60633`, the 28-day CETES.
- The dashboard headline showed its dollar signs as a formula; they are now escaped.

### Known limitations

- MLflow tracking is not implemented yet.
- The Docker base image is not pinned by digest.
- The hybrid model carries exogenous variables forward over the horizon.
- The gate uses a single validation window and `gate_min_improvement` is 0 by default, so it can accept a tree that does not generalize. Only rejections are logged.
- When the gate rejects the tree, the forecast of that target ignores the scenario shocks. The Technical details tab of the dashboard explains it.
- The model comparison needs `reports/metrics.json`, which is not versioned and is not mounted in the Docker `app` service. Without it the tab explains how to create the file.
- The dashboard text is in English only, and the layout was checked in a desktop browser only.
- `make ingest` stops at the end date of `configs/simulator.yaml`; pass `--end` to bring the data up to date. With `DB_URL` set it stops if the database is unreachable.
