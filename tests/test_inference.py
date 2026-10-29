"""Tests for inference with lightweight fake models."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.feature_engineering import add_calendar_features
from fxhedge.inference import (
    OUTPUT_COLUMNS,
    apply_scenario,
    interval_bounds,
    log_change_samples,
    predict_forecast,
    validate_request,
)
from fxhedge.model_training import Target, TrainedModel

FEATURES = ("cetes_28d", "is_weekend")


class ConstBase:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), self.value)

    def to_json(self) -> str:
        return "{}"


class CoefTree:
    def __init__(self, coef: tuple[float, ...]) -> None:
        self.coef = np.asarray(coef)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features.to_numpy(dtype=np.float64) @ self.coef

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


def _model(target: Target, base: float, coef: tuple[float, ...]) -> TrainedModel:
    return TrainedModel(
        kind="hybrid",
        target=target,
        base=ConstBase(base),
        tree=CoefTree(coef),
        feature_columns=FEATURES,
        params={},
        train_end=pd.Timestamp("2024-06-30"),
    )


def _history(days: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    rng = np.random.default_rng(1)
    frame["cetes_28d"] = 10.0
    frame["y_fx"] = 17.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, days)))
    frame["y_cash"] = 100_000.0 * np.exp(rng.normal(0.0, 0.05, days))
    return frame


def _models() -> dict[Target, TrainedModel]:
    return {
        "y_fx": _model("y_fx", 17.0, (0.01, 0.0)),
        "y_cash": _model("y_cash", 90_000.0, (100.0, 0.0)),
    }


def test_forecast_shape_columns_and_dates() -> None:
    result = predict_forecast(_models(), _history(), date(2024, 6, 1), 30)
    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert len(result) == 30
    assert result["date"].iloc[0] == pd.Timestamp("2024-06-02")
    assert result["date"].iloc[-1] == pd.Timestamp("2024-07-01")


def test_intervals_contain_point_forecast() -> None:
    result = predict_forecast(_models(), _history(), date(2024, 6, 1), 60)
    assert (result["fx_lower"] < result["fx_predicted"]).all()
    assert (result["fx_predicted"] < result["fx_upper"]).all()
    assert (result["cash_lower"] < result["cash_expected_usd"]).all()
    assert (result["cash_expected_usd"] < result["cash_upper"]).all()


def test_scenarios_shock_exogenous_features() -> None:
    base = predict_forecast(_models(), _history(), date(2024, 6, 1), 30, "base")
    up = predict_forecast(_models(), _history(), date(2024, 6, 1), 30, "alza")
    down = predict_forecast(_models(), _history(), date(2024, 6, 1), 30, "baja")
    assert (up["fx_predicted"] > base["fx_predicted"]).all()
    assert (down["fx_predicted"] < base["fx_predicted"]).all()
    assert up["fx_predicted"].iloc[0] - base["fx_predicted"].iloc[0] == pytest.approx(0.01)


def test_apply_scenario_leaves_calendar_columns_untouched() -> None:
    features = pd.DataFrame({"cetes_28d": [10.0], "month": [6]})
    shocked = apply_scenario(features, "alza")
    assert shocked["cetes_28d"].iloc[0] == pytest.approx(11.0)
    assert shocked["month"].iloc[0] == 6


def test_provided_error_samples_define_the_interval() -> None:
    errors = {"y_fx": np.array([-0.1, 0.0, 0.1] * 50), "y_cash": np.array([-0.2, 0.0, 0.2] * 50)}
    result = predict_forecast(_models(), _history(), date(2024, 6, 1), 30, error_samples=errors)
    ratio = result["fx_upper"] / result["fx_predicted"]
    assert ratio.iloc[0] == pytest.approx(np.exp(0.1), rel=1e-6)


@pytest.mark.parametrize(
    ("horizon", "scenario"), [(45, "base"), (30, "extreme")], ids=["horizon", "scenario"]
)
def test_validate_request_rejects_unsupported_values(horizon: int, scenario: str) -> None:
    with pytest.raises(ValueError):
        validate_request(horizon, scenario)


def test_forecast_requires_history_before_run_date() -> None:
    with pytest.raises(ValueError):
        predict_forecast(_models(), _history(), date(2000, 1, 1), 30)


def test_log_change_samples_are_centered_and_validated() -> None:
    samples = log_change_samples(_history()["y_fx"], 30)
    assert np.median(samples) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        log_change_samples(pd.Series([1.0, 2.0, 3.0]), 30)


def test_interval_bounds_validation() -> None:
    point = np.array([10.0, 20.0])
    lower, upper = interval_bounds(point, np.array([-0.1, 0.0, 0.1] * 20), 0.9)
    assert (lower < point).all()
    assert (upper > point).all()
    with pytest.raises(ValueError):
        interval_bounds(point, np.array([]), 0.9)
    with pytest.raises(ValueError):
        interval_bounds(point, np.array([0.0]), 1.5)
