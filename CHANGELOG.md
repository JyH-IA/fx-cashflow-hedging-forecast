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

### Known limitations

- MLflow tracking is not implemented yet.
- The Docker base image is not pinned by digest.
- The hybrid model carries exogenous variables forward over the horizon.
