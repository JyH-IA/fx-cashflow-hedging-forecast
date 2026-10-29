.PHONY: setup lint typecheck test db-up ingest features train evaluate hedge explain monitor retrain app api docker-build

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

test:
	pytest

db-up:
	docker compose up -d postgres

ingest:
	$(PYTHON) -m fxhedge.data_ingestion

features:
	$(PYTHON) -m fxhedge.feature_engineering

train:
	$(PYTHON) -m fxhedge.model_training

evaluate:
	$(PYTHON) -m fxhedge.evaluation

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

docker-build:
	docker build -t fxhedge:local .
