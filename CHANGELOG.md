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

### Changed

- The hybrid for `y_fx` uses a random walk as its base instead of Prophet, which lost to a random walk at every origin of the diagnostic. `y_cash` keeps Prophet. The base is chosen per target in `BASE_BY_TARGET`.
- Residuals of the random-walk base are computed with no gap, because that base is refitted up to each forecast origin.

### Known limitations

- MLflow tracking is not implemented yet.
- The Docker base image is not pinned by digest.
- The hybrid model carries exogenous variables forward over the horizon.
- The gate uses a single validation window and `gate_min_improvement` is 0 by default, so it can accept a tree that does not generalize. Only rejections are logged.
- When the gate rejects the tree, the forecast of that target ignores the scenario shocks. The dashboard shows a notice about it.
