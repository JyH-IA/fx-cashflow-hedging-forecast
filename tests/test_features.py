"""Tests for feature engineering."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.config import SimulatorConfig
from fxhedge.feature_engineering import (
    add_calendar_features,
    add_lags_and_rollings,
    align_series,
    build_master_features,
)
from fxhedge.io_utils import write_csv
from fxhedge.schemas import DataContractError
from fxhedge.simulator import simulate_internal_sales


def _macro(start: str, days: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    rng = np.random.default_rng(0)
    fix = 17.0 + np.cumsum(rng.normal(0.0, 0.05, days))
    frame = pd.DataFrame(
        {
            "date": dates,
            "usdmxn_fix": fix,
            "usdmxn_yf": fix + 0.01,
            "cetes_28d": 10.0,
            "fed_funds": np.nan,
            "inpc": 130.0,
            "cancun_arrivals": np.nan,
        }
    )
    frame.loc[dates.dayofweek >= 5, ["usdmxn_fix", "usdmxn_yf"]] = np.nan
    return frame.astype({"fed_funds": float, "cancun_arrivals": float})


def _sales(start: str, days: int) -> pd.DataFrame:
    cfg = SimulatorConfig(start=date(2024, 1, 1), end=date(2024, 12, 31))
    first = pd.Timestamp(start).date()
    last = (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).date()
    return simulate_internal_sales(cfg, first, last, seed=1)


def test_align_series_inner_join_and_forward_fill() -> None:
    macro = _macro("2024-01-01", 20)
    sales = _sales("2024-01-05", 10)
    aligned = align_series(macro, sales)
    assert aligned["date"].min() == pd.Timestamp("2024-01-05")
    assert aligned["date"].max() == pd.Timestamp("2024-01-14")
    assert aligned["y_fx"].notna().all()
    assert (aligned["y_cash"] == aligned["cash_in_usd"]).all()


def test_align_series_rejects_invalid_contract() -> None:
    macro = _macro("2024-01-01", 5).drop(columns=["inpc"])
    with pytest.raises(DataContractError):
        align_series(macro, _sales("2024-01-01", 5))


def test_lags_and_rollings_do_not_leak_current_value() -> None:
    df = pd.DataFrame({"x": np.arange(1.0, 11.0)})
    out = add_lags_and_rollings(df, ["x"], lags=[1, 2], windows=[3])
    assert out["x_lag1"].iloc[3] == 3.0
    assert out["x_lag2"].iloc[3] == 2.0
    assert out["x_roll3_mean"].iloc[3] == pytest.approx(2.0)
    assert np.isnan(out["x_roll3_mean"].iloc[2])


@pytest.mark.parametrize(
    ("lags", "windows"), [([0], [3]), ([1], [1])], ids=["zero-lag", "window-one"]
)
def test_lags_and_rollings_validate_arguments(lags: list[int], windows: list[int]) -> None:
    df = pd.DataFrame({"x": np.arange(5.0)})
    with pytest.raises(ValueError):
        add_lags_and_rollings(df, ["x"], lags, windows)


def test_lags_and_rollings_reject_missing_column() -> None:
    with pytest.raises(ValueError):
        add_lags_and_rollings(pd.DataFrame({"x": [1.0, 2.0]}), ["y"], [1], [2])


def test_calendar_features_flags() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-06", "2024-09-15"])})
    out = add_calendar_features(df)
    assert out["is_holiday_mx"].tolist()[0] == 1
    assert out["is_weekend"].tolist() == [0, 1, 1]
    assert out["is_high_season"].tolist() == [1, 1, 0]
    assert out["is_hurricane_season"].tolist() == [0, 0, 1]
    assert out["quarter"].tolist() == [1, 1, 3]


def test_build_master_features_end_to_end(tmp_path: Path) -> None:
    macro_path = tmp_path / "macro.csv"
    sales_path = tmp_path / "sales.csv"
    out_path = tmp_path / "processed" / "master.csv"
    write_csv(_macro("2024-01-01", 120), macro_path)
    write_csv(_sales("2024-01-01", 120), sales_path)

    master = build_master_features(macro_path, sales_path, out_path)

    assert out_path.exists()
    assert not master.isna().any().any()
    assert len(master) == 120 - 30
    assert {"y_fx", "y_cash", "y_fx_lag7", "y_cash_roll30_mean", "is_holiday_mx"} <= set(
        master.columns
    )
    assert "fed_funds" not in master.columns
    for leaky in ("usdmxn_fix", "sales_usd", "cash_in_usd", "occupancy_rate"):
        assert leaky not in master.columns


def test_build_master_features_needs_enough_rows(tmp_path: Path) -> None:
    write_csv(_macro("2024-01-01", 20), tmp_path / "m.csv")
    write_csv(_sales("2024-01-01", 20), tmp_path / "s.csv")
    with pytest.raises(ValueError):
        build_master_features(tmp_path / "m.csv", tmp_path / "s.csv", tmp_path / "o.csv")
