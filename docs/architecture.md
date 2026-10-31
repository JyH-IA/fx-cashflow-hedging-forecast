# Architecture

## Data flow

1. **Ingestion** (`data_ingestion`): Banxico SIE series and the Yahoo Finance close are downloaded with timeouts, retries and response validation. The Fed funds rate and, optionally, SECTUR arrivals are loaded from manual CSVs. Internal sales come from the simulator, optionally stored in a SQL database when `DB_URL` is set. `make sample` writes fully synthetic data instead. Both commands record the source in `data/raw/data_source.txt`.
2. **Features** (`feature_engineering`): inner join by date, forward fill for non-banking days, lags and rolling statistics on shifted series, and calendar flags (weekday, Mexican holidays, high season, hurricane season). Exogenous columns that are entirely empty are dropped with a warning. The source marker is copied to `data/processed/`.
3. **Training** (`model_training`): each hybrid has a base forecast (Prophet for cash-in, the last observed value for the exchange rate) plus a LightGBM model, tuned with Optuna, that learns the out-of-sample residuals of the base. A validation gate keeps the LightGBM part only if it lowers the error on the most recent residuals, scored with exogenous variables frozen at the forecast origin. XGBoost predicts the level directly and acts as a comparator.
4. **Evaluation** (`evaluation`, `hedging_backtest`): rolling-origin backtests against baselines, then a historical simulation of hedging strategies.
5. **Serving** (`inference`, `api`, `dashboard`, `app/streamlit_app.py`): forecasts with scenarios and intervals, exposed through FastAPI and a Streamlit dashboard with four tabs (Summary, Hedging, How reliable is it?, Technical details). A line under the title shows the data source, the date of the last observation and the last USD/MXN rate.
6. **Monitoring** (`monitoring`): PSI per variable, rolling error and a retraining rule.

## Modules

| Module | Responsibility |
|---|---|
| `config` | Environment settings (`SecretStr`) and validated YAML models |
| `schemas` | Data contracts and `validate_frame` |
| `db` | Parameterized SQL access with SQLAlchemy |
| `simulator`, `sample_data` | Synthetic sales and fully synthetic macro data |
| `data_source` | Marker file that records whether the raw data is synthetic or from Banxico |
| `finance` | Forward pricing and hedge proceeds |
| `dashboard` | Pure helpers behind the Streamlit app: headline, key figures, labels and the data source line |
| `explain` | SHAP values and a driver summary (skipped for a target whose tree the gate rejected) |
| `io_utils` | CSV helpers and SHA-256 hashing |

## API

The FastAPI service exposes `GET /forecast` and `GET /health`. The contract can be exported with `make openapi`; `docs/openapi.json` is not versioned. The service binds to `127.0.0.1`; put it behind authentication or a TLS reverse proxy before exposing it. There is no client example.

## Operational notes

- Models and data are mounted into the containers at runtime; the image contains neither.
- Each model directory holds `metadata.json` with the base (`base_model`), whether the residual tree is used (`residual_model`), the gate's holdout errors (`gate`) and the SHA-256 of every artifact. A rejected tree has no `lightgbm_model.txt`.
- The scheduled workflow `monitoring.yml` writes a drift report from synthetic data and uploads it as an artifact.
