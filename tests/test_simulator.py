"""Tests for the synthetic sales simulator."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fxhedge.config import SimulatorConfig
from fxhedge.schemas import SALES_SPEC, validate_frame
from fxhedge.simulator import simulate_internal_sales

CFG = SimulatorConfig(start=date(2020, 1, 1), end=date(2025, 12, 31))


def test_same_seed_gives_identical_data_and_other_seed_differs() -> None:
    first = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    again = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    other = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=2)
    assert first.equals(again)
    assert not first.equals(other)


def test_extending_the_end_date_keeps_the_earlier_days() -> None:
    short = simulate_internal_sales(CFG, CFG.start, date(2024, 12, 31), seed=1)
    longer = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    pd.testing.assert_frame_equal(longer.iloc[: len(short)].reset_index(drop=True), short)


def test_output_satisfies_the_sales_contract() -> None:
    frame = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    assert validate_frame(frame, SALES_SPEC) is frame
    assert len(frame) == (CFG.end - CFG.start).days + 1


def test_caribbean_seasonality_peaks_in_spring_and_dips_in_hurricane_season() -> None:
    frame = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    monthly = frame.groupby(frame["date"].dt.month)["occupancy_rate"].mean()
    assert monthly[3] > monthly[9] + 0.2
    assert monthly[9] == monthly.min()


def test_sales_grow_over_the_years() -> None:
    frame = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    yearly = frame.groupby(frame["date"].dt.year)["sales_usd"].mean()
    assert yearly[2025] > yearly[2020]


def test_cash_in_is_a_fraction_of_sales() -> None:
    frame = simulate_internal_sales(CFG, CFG.start, CFG.end, seed=1)
    ratio = (frame["cash_in_usd"] / frame["sales_usd"]).mean()
    assert ratio == pytest.approx(CFG.cash_conversion_ratio, rel=0.05)


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_internal_sales(CFG, date(2024, 2, 1), date(2024, 1, 1), seed=1)
