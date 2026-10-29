"""Tests for the dashboard helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxhedge.config import BusinessConfig
from fxhedge.dashboard import (
    DEFAULT_RATIOS,
    commission_saving_mxn,
    expected_cash_in_usd,
    forecast_summary,
    format_mxn,
    impact_table,
)

BUSINESS = BusinessConfig()


def _frame(days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-11-01", periods=days, freq="D")
    base = np.linspace(17.0, 18.0, days)
    return pd.DataFrame(
        {
            "date": dates,
            "fx_predicted": base,
            "fx_lower": base * 0.95,
            "fx_upper": base * 1.05,
            "cash_expected_usd": np.full(days, 10_000.0),
        }
    )


def test_expected_cash_in_sums_the_horizon() -> None:
    assert expected_cash_in_usd(_frame()) == pytest.approx(300_000.0)


def test_forecast_summary_reports_settlement_and_change() -> None:
    summary = forecast_summary(_frame(), spot=17.0)
    assert summary["settlement_fx"] == pytest.approx(18.0)
    assert summary["settlement_lower"] == pytest.approx(18.0 * 0.95)
    assert summary["change_pct"] == pytest.approx((18.0 / 17.0 - 1.0) * 100.0)
    assert summary["expected_cash_usd"] == pytest.approx(300_000.0)


def test_impact_table_range_narrows_as_hedge_ratio_rises() -> None:
    table = impact_table(_frame(), 17.0, 300_000.0, DEFAULT_RATIOS, BUSINESS, 30)
    assert table["hedge_ratio"].tolist() == list(DEFAULT_RATIOS)
    assert table["range_mxn"].is_monotonic_decreasing
    assert table["range_mxn"].iloc[-1] < 0.05 * table["range_mxn"].iloc[0]
    assert (table["worst_case_mxn"] <= table["best_case_mxn"]).all()


def test_unhedged_row_has_no_bank_cost_and_zero_benefit() -> None:
    table = impact_table(_frame(), 17.0, 300_000.0, [0.0, 1.0], BUSINESS, 30)
    assert table["bank_cost_mxn"].iloc[0] == pytest.approx(0.0)
    assert table["expected_benefit_mxn"].iloc[0] == pytest.approx(0.0)
    assert table["bank_cost_mxn"].iloc[1] > 0.0


def test_impact_table_validates_inputs() -> None:
    with pytest.raises(ValueError):
        impact_table(_frame(), 17.0, -1.0, [0.5], BUSINESS, 30)
    with pytest.raises(ValueError):
        impact_table(_frame().drop(columns=["fx_lower"]), 17.0, 1.0, [0.5], BUSINESS, 30)
    with pytest.raises(ValueError):
        impact_table(_frame().iloc[0:0], 17.0, 1.0, [0.5], BUSINESS, 30)
    with pytest.raises(ValueError):
        forecast_summary(_frame(), spot=0.0)


def test_commission_saving_grows_as_less_is_hedged() -> None:
    full = commission_saving_mxn(300_000.0, 1.0, 17.0, BUSINESS, 30)
    half = commission_saving_mxn(300_000.0, 0.5, 17.0, BUSINESS, 30)
    none = commission_saving_mxn(300_000.0, 0.0, 17.0, BUSINESS, 30)
    assert full == pytest.approx(0.0)
    assert 0.0 < half < none


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1234567.4, "$1,234,567 MXN"), (-2500.0, "-$2,500 MXN"), (0.0, "$0 MXN")],
)
def test_format_mxn(value: float, expected: str) -> None:
    assert format_mxn(value) == expected
