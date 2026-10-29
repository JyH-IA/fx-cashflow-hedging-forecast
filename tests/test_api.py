"""Tests for the FastAPI application (skipped when FastAPI is not installed)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fxhedge.api import create_app  # noqa: E402
from fxhedge.api_models import ScenarioName  # noqa: E402
from fxhedge.model_training import ModelIntegrityError  # noqa: E402


def _forecast(run_date: date, horizon: int, scenario: ScenarioName) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(run_date) + pd.Timedelta(days=1), periods=horizon)
    base = np.full(horizon, 17.5)
    return pd.DataFrame(
        {
            "date": dates,
            "fx_predicted": base,
            "cash_expected_usd": base * 1000.0,
            "fx_lower": base * 0.95,
            "fx_upper": base * 1.05,
            "cash_lower": base * 900.0,
            "cash_upper": base * 1100.0,
        }
    )


def _client() -> TestClient:
    return TestClient(create_app(_forecast))


def test_health_endpoint() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_endpoint_returns_requested_horizon() -> None:
    response = _client().get(
        "/forecast", params={"horizon": 60, "scenario": "alza", "run_date": "2026-10-29"}
    )
    body = response.json()
    assert response.status_code == 200
    assert body["horizon"] == 60
    assert body["scenario"] == "alza"
    assert len(body["points"]) == 60
    assert "not financial advice" in body["disclaimer"]


@pytest.mark.parametrize(
    "params",
    [{"horizon": 45}, {"scenario": "extreme"}, {"run_date": "not-a-date"}],
    ids=["horizon", "scenario", "date"],
)
def test_invalid_parameters_are_rejected(params: dict[str, object]) -> None:
    assert _client().get("/forecast", params=params).status_code == 422


def test_missing_models_return_503_without_internal_details() -> None:
    def missing(run_date: date, horizon: int, scenario: ScenarioName) -> pd.DataFrame:
        raise FileNotFoundError("/secret/path/models")

    response = TestClient(create_app(missing)).get("/forecast")
    assert response.status_code == 503
    assert "secret" not in response.text


def test_integrity_failure_returns_503() -> None:
    def tampered(run_date: date, horizon: int, scenario: ScenarioName) -> pd.DataFrame:
        raise ModelIntegrityError("hash mismatch for prophet_model.json")

    response = TestClient(create_app(tampered)).get("/forecast")
    assert response.status_code == 503
    assert "hash" not in response.text


def test_openapi_documents_the_forecast_contract() -> None:
    schema = _client().get("/openapi.json").json()
    assert "/forecast" in schema["paths"]
    assert "ForecastResponse" in schema["components"]["schemas"]
