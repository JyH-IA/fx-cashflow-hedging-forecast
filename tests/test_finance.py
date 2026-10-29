"""Tests for hedging finance."""

from __future__ import annotations

import pytest

from fxhedge.finance import (
    compute_hedge_savings,
    forward_points,
    forward_rate,
    hedged_proceeds_mxn,
)


def test_forward_rate_matches_covered_interest_parity() -> None:
    expected = 17.0 * (1.0 + 0.10 * 90.0 / 360.0) / (1.0 + 0.045 * 90.0 / 360.0)
    assert forward_rate(17.0, 10.0, 4.5, 90) == pytest.approx(expected)
    assert forward_points(17.0, 10.0, 4.5, 90) == pytest.approx(expected - 17.0)


def test_forward_equals_spot_when_rates_are_equal() -> None:
    assert forward_rate(18.0, 5.0, 5.0, 60) == pytest.approx(18.0)


@pytest.mark.parametrize(
    ("spot", "mxn", "usd", "days"),
    [(0.0, 10.0, 4.0, 30), (17.0, -1.0, 4.0, 30), (17.0, 10.0, 4.0, 0)],
    ids=["zero-spot", "negative-rate", "zero-days"],
)
def test_forward_rate_validates_inputs(spot: float, mxn: float, usd: float, days: int) -> None:
    with pytest.raises(ValueError):
        forward_rate(spot, mxn, usd, days)


def test_unhedged_proceeds_equal_notional_times_spot() -> None:
    assert hedged_proceeds_mxn(1000.0, 0.0, 17.5, 16.0, 15.0, 25.0) == pytest.approx(16000.0)


def test_fully_hedged_proceeds_use_forward_minus_costs() -> None:
    proceeds = hedged_proceeds_mxn(1000.0, 1.0, 17.5, 16.0, 10.0, 25.0)
    assert proceeds == pytest.approx(1000.0 * 17.5 * (1 - 0.001) - 25.0 * 16.0)


def test_hedged_proceeds_validate_ratio_and_costs() -> None:
    with pytest.raises(ValueError):
        hedged_proceeds_mxn(1000.0, 1.5, 17.5, 16.0, 10.0, 25.0)
    with pytest.raises(ValueError):
        hedged_proceeds_mxn(1000.0, 0.5, 17.5, 16.0, -1.0, 25.0)


def test_savings_positive_when_forward_beats_forecast_spot() -> None:
    result = compute_hedge_savings(100_000.0, 0.5, 17.0, 17.5, 10.0, 25.0)
    assert result.hedged_notional_usd == pytest.approx(50_000.0)
    assert result.expected_benefit_mxn > 0.0
    assert result.expected_benefit_usd == pytest.approx(result.expected_benefit_mxn / 17.0)


def test_savings_fall_with_wider_spread() -> None:
    narrow = compute_hedge_savings(100_000.0, 0.5, 17.0, 17.5, 5.0, 25.0)
    wide = compute_hedge_savings(100_000.0, 0.5, 17.0, 17.5, 50.0, 25.0)
    assert wide.expected_benefit_mxn < narrow.expected_benefit_mxn
    assert wide.bank_cost_mxn > narrow.bank_cost_mxn


def test_no_hedge_has_no_cost_or_benefit() -> None:
    result = compute_hedge_savings(100_000.0, 0.0, 17.0, 17.5, 10.0, 25.0)
    assert result.expected_benefit_mxn == pytest.approx(0.0)
    assert result.bank_cost_mxn == pytest.approx(0.0)
