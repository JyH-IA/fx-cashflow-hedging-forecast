.PHONY: setup format lint typecheck test sample demo db-up ingest features train evaluate evaluate-full hedge explain monitor retrain app api openapi docker-build

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

# Fully synthetic data, no token needed (the exchange-rate series is NOT real market data).
sample:
	$(PYTHON) -m fxhedge.sample_data

# End-to-end demo on synthetic data.
demo: sample features train evaluate hedge

db-up:
	docker compose up -d postgres

# Real macro data; needs BANXICO_TOKEN in .env.
ingest:
	$(PYTHON) -m fxhedge.data_ingestion

features:
	$(PYTHON) -m fxhedge.feature_engineering

train:
	$(PYTHON) -m fxhedge.model_training

evaluate:
	$(PYTHON) -m fxhedge.evaluation

# Also backtests the hybrid, XGBoost and SARIMA. Refits per fold, so it is slow.
evaluate-full:
	$(PYTHON) -m fxhedge.evaluation --with-models --with-sarima

hedge:
	$(PYTHON) -m fxhedge.hedging_backtest

explain:
	$(PYTHON) -m fxhedge.explain

monitor:
	$(PYTHON) -m fxhedge.monitoring

retrain:
	$(PYTHON) -m fxhedge.monitoring --retrain

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn fxhedge.api:app --host 127.0.0.1 --port 8000

openapi:
	$(PYTHON) -m fxhedge.api --export-openapi docs/openapi.json

docker-build:
	docker build -t fxhedge:local .
