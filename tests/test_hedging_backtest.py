"""Tests for the historical hedging simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.config import BusinessConfig
from fxhedge.hedging_backtest import (
    HedgeConfig,
    compare_strategies,
    simulate_strategy,
    write_hedging_report,
)

CFG = HedgeConfig(
    horizon_days=30,
    bank_spread_bps=0.0,
    fixed_fee_usd=0.0,
    rate_mxn_pct=0.0,
    rate_usd_pct=0.0,
)


def _data(days: int = 720, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    fx = 17.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, days)))
    cash = 100_000.0 + rng.normal(0.0, 2_000.0, days)
    return pd.DataFrame({"date": dates, "y_fx": fx, "y_cash": cash})


def test_from_business_copies_assumptions() -> None:
    cfg = HedgeConfig.from_business(BusinessConfig(), horizon_days=60)
    assert cfg.horizon_days == 60
    assert cfg.fixed_ratio == 0.5
    assert cfg.mxn_cost_share == 0.82


def test_simulation_has_one_row_per_month_and_correct_ratios() -> None:
    data = _data()
    none = simulate_strategy(data, "none", CFG)
    fixed = simulate_strategy(data, "fixed", CFG)
    assert len(none) >= 22
    assert (none["hedge_ratio"] == 0.0).all()
    assert (fixed["hedge_ratio"] == 0.5).all()
    assert none["origin"].equals(fixed["origin"])
    assert (none["settle_date"] - none["origin"]).dt.days.eq(30).all()


def test_fully_hedged_margin_is_locked_in_without_costs() -> None:
    cfg = HedgeConfig(
        horizon_days=30,
        fixed_ratio=1.0,
        bank_spread_bps=0.0,
        fixed_fee_usd=0.0,
        rate_mxn_pct=0.0,
        rate_usd_pct=0.0,
    )
    table = simulate_strategy(_data(), "fixed", cfg)
    locked = table["exposure_usd"] * table["spot0"] * (1.0 - cfg.mxn_cost_share)
    assert np.allclose(table["margin_mxn"], locked)


def test_forecast_strategy_switches_ratio_with_the_forecast() -> None:
    data = _data()
    origins = simulate_strategy(data, "none", CFG)["origin"]
    forward = simulate_strategy(data, "none", CFG)["forward"].to_numpy()
    alternating = np.where(np.arange(len(origins)) % 2 == 0, forward * 0.9, forward * 1.1)
    forecast = pd.Series(alternating, index=origins)
    table = simulate_strategy(data, "forecast", CFG, forecast)
    assert set(table["hedge_ratio"]) == {CFG.forecast_high_ratio, CFG.forecast_low_ratio}
    assert table["hedge_ratio"].iloc[0] == CFG.forecast_high_ratio
    assert table["hedge_ratio"].iloc[1] == CFG.forecast_low_ratio


def test_forecast_strategy_requires_forecasts() -> None:
    with pytest.raises(ValueError):
        simulate_strategy(_data(), "forecast", CFG)


def test_simulation_validates_columns_and_length() -> None:
    with pytest.raises(ValueError):
        simulate_strategy(_data().drop(columns=["y_cash"]), "none", CFG)
    with pytest.raises(ValueError):
        simulate_strategy(_data(20), "none", CFG)


def test_hedging_reduces_margin_variance_and_bootstrap_is_reproducible() -> None:
    data = _data()
    results = {name: simulate_strategy(data, name, CFG) for name in ("none", "fixed")}
    first = compare_strategies(results, n_boot=200, block=3, seed=1).summary
    second = compare_strategies(results, n_boot=200, block=3, seed=1).summary
    assert first.equals(second)
    row = first.set_index("strategy").loc["fixed"]
    assert row["var_reduction"] > 0.3
    assert row["var_reduction_lo"] <= row["var_reduction"] <= row["var_reduction_hi"] + 1e-9
    base = first.set_index("strategy").loc["none"]
    assert base["var_reduction"] == pytest.approx(0.0)


def test_compare_requires_baseline_and_matching_origins() -> None:
    data = _data()
    fixed = simulate_strategy(data, "fixed", CFG)
    with pytest.raises(ValueError):
        compare_strategies({"fixed": fixed}, 10, 3, 0)
    shifted = simulate_strategy(_data(500), "none", CFG)
    with pytest.raises(ValueError):
        compare_strategies({"none": shifted, "fixed": fixed}, 10, 3, 0)


def test_report_is_written(tmp_path: Path) -> None:
    data = _data()
    results = {name: simulate_strategy(data, name, CFG) for name in ("none", "fixed")}
    comparison = compare_strategies(results, n_boot=20, block=3, seed=0)
    out = tmp_path / "reports" / "hedging.md"
    write_hedging_report(comparison, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Historical hedging simulation")
    assert "not financial advice" in text
