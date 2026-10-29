# Architecture

## Data flow

1. **Ingestion** (`data_ingestion`): Banxico SIE series and the Yahoo Finance close are downloaded with timeouts, retries and response validation. SECTUR arrivals and the Fed funds rate are loaded from manual CSVs. Internal sales come from the simulator, optionally stored in a SQL database.
2. **Features** (`feature_engineering`): inner join by date, forward fill for non-banking days, lags and rolling statistics on shifted series, and calendar flags (weekday, Mexican holidays, high season, hurricane season).
3. **Training** (`model_training`): Prophet models trend and seasonality; LightGBM, tuned with Optuna, learns the out-of-sample Prophet residuals. XGBoost predicts the level directly and acts as a comparator.
4. **Evaluation** (`evaluation`, `hedging_backtest`): rolling-origin backtests against baselines, then a historical simulation of hedging strategies.
5. **Serving** (`inference`, `api`, `dashboard`, `app/streamlit_app.py`): forecasts with scenarios and intervals, exposed through FastAPI and Streamlit.
6. **Monitoring** (`monitoring`): PSI per variable, rolling error and a retraining rule.

## Modules

| Module | Responsibility |
|---|---|
| `config` | Environment settings (`SecretStr`) and validated YAML models |
| `schemas` | Data contracts and `validate_frame` |
| `db` | Parameterized SQL access with SQLAlchemy |
| `simulator`, `sample_data` | Synthetic sales and fully synthetic macro data |
| `finance` | Forward pricing and hedge proceeds |
| `explain` | SHAP values and a driver summary |
| `io_utils` | CSV helpers and SHA-256 hashing |

## Integration with a Java backend

The FastAPI service exposes `GET /forecast` and `GET /health`. The contract can be exported with `make openapi` (`docs/openapi.json`) and used to generate a typed client for Spring Boot. The service binds to `127.0.0.1`; put it behind authentication or a TLS reverse proxy before exposing it. A sample Java client is not included yet.

## Operational notes

- Models and data are mounted into the containers at runtime; the image contains neither.
- The scheduled workflow `monitoring.yml` writes a drift report from synthetic data and uploads it as an artifact.
