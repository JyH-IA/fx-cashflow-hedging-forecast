"""Tests for the fully synthetic sample data and the offline pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fxhedge import sample_data
from fxhedge.config import SimulatorConfig
from fxhedge.feature_engineering import build_master_features
from fxhedge.sample_data import simulate_macro
from fxhedge.schemas import MACRO_SPEC, validate_frame
from fxhedge.simulator import simulate_internal_sales


def test_macro_satisfies_its_contract_and_is_reproducible() -> None:
    first = simulate_macro(date(2022, 1, 1), date(2023, 12, 31), seed=7)
    again = simulate_macro(date(2022, 1, 1), date(2023, 12, 31), seed=7)
    assert validate_frame(first, MACRO_SPEC) is first
    assert first.equals(again)


def test_exchange_rates_are_missing_on_weekends_only() -> None:
    frame = simulate_macro(date(2024, 1, 1), date(2024, 1, 31), seed=1)
    weekend = frame["date"].dt.dayofweek >= 5
    assert frame.loc[weekend, "usdmxn_fix"].isna().all()
    assert frame.loc[~weekend, "usdmxn_fix"].notna().all()


def test_inflation_is_constant_within_a_month_and_rises_across_months() -> None:
    frame = simulate_macro(date(2024, 1, 1), date(2024, 3, 31), seed=1)
    monthly = frame.groupby(frame["date"].dt.month)["inpc"]
    assert (monthly.nunique() == 1).all()
    assert monthly.first().is_monotonic_increasing


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_macro(date(2024, 2, 1), date(2024, 1, 1), seed=1)


def test_offline_pipeline_from_sample_data_to_master_features(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONFIGS_DIR", str(repo_root / "configs"))
    code = sample_data.main(["--start", "2022-01-01", "--end", "2023-06-30"])
    assert code == 0
    raw = tmp_path / "data" / "raw"
    master = build_master_features(
        raw / "macro_data.csv", raw / "internal_sales_simulated.csv", tmp_path / "master.csv"
    )
    assert not master.isna().any().any()
    assert {"y_fx", "y_cash", "cetes_28d", "is_weekend"} <= set(master.columns)
    assert len(master) > 400


def test_sales_and_macro_share_the_same_calendar() -> None:
    cfg = SimulatorConfig(start=date(2024, 1, 1), end=date(2024, 2, 29))
    sales = simulate_internal_sales(cfg, cfg.start, cfg.end, seed=1)
    macro = simulate_macro(cfg.start, cfg.end, seed=1)
    assert sales["date"].equals(macro["date"])
