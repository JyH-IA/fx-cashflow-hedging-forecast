"""Tests for the API response models."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fxhedge.api_models import DISCLAIMER, build_response


def _frame(days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-11-01", periods=days, freq="D")
    base = np.linspace(17.0, 18.0, days)
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


def test_build_response_maps_every_row() -> None:
    response = build_response(_frame(), date(2026, 10, 31), 30, "base")
    assert response.horizon == 30
    assert response.scenario == "base"
    assert response.disclaimer == DISCLAIMER
    assert len(response.points) == 30
    assert response.points[0].date == date(2026, 11, 1)
    assert response.points[-1].fx_predicted == pytest.approx(18.0)


def test_build_response_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        build_response(_frame().drop(columns=["fx_lower"]), date(2026, 10, 31), 30, "base")


def test_build_response_rejects_unsupported_horizon() -> None:
    with pytest.raises(ValueError):
        build_response(_frame(45), date(2026, 10, 31), 45, "base")


def test_build_response_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        build_response(_frame(10), date(2026, 10, 31), 30, "base")


def test_build_response_rejects_non_finite_values() -> None:
    frame = _frame()
    frame.loc[3, "fx_predicted"] = np.nan
    with pytest.raises(ValueError):
        build_response(frame, date(2026, 10, 31), 30, "base")


def test_response_serializes_to_json_with_iso_dates() -> None:
    response = build_response(_frame(), date(2026, 10, 31), 30, "alza")
    text = response.model_dump_json()
    assert '"run_date":"2026-10-31"' in text
    assert '"scenario":"alza"' in text
