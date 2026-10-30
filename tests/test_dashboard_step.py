"""Tests for the step of the amount-to-protect slider in the dashboard."""

from __future__ import annotations

import pytest

from fxhedge.dashboard import notional_step


@pytest.mark.parametrize(
    ("expected", "step"),
    [(0.0, 1.0), (50.0, 1.0), (149.0, 1.0), (12_345.0, 123.0), (1_000_000.0, 10_000.0)],
)
def test_step_is_about_one_percent_of_the_inflow_and_at_least_one(
    expected: float, step: float
) -> None:
    assert notional_step(expected) == pytest.approx(step)


def test_step_is_a_float_like_the_other_slider_arguments() -> None:
    assert isinstance(notional_step(12_345.0), float)
    assert isinstance(notional_step(0.0), float)
