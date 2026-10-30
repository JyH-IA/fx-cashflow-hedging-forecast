"""Tests for the random-walk base used by the y_fx hybrid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge import model_training as mt
from fxhedge.config import TrainingConfig
from fxhedge.feature_engineering import add_calendar_features
from fxhedge.model_training import (
    BaseForecaster,
    LastValueBase,
    Predictor,
    ZeroResidual,
    fit_hybrid,
    fit_last_value_base,
    load_last_value_base,
    load_trained_model,
    oos_residuals,
    save_model,
)

CFG = TrainingConfig(n_splits=3, gap=5, optuna_trials=1)


class MeanBase:
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), self.mean, dtype=np.float64)

    def to_json(self) -> str:
        return json.dumps({"mean": self.mean})


def fit_mean_base(series: pd.Series) -> BaseForecaster:
    return MeanBase(float(series.mean()))


def fit_zero(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    return ZeroResidual(), {}


def _frame(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    frame["level"] = np.arange(days, dtype=np.float64)
    frame["y_fx"] = 17.0 + 0.001 * np.arange(days, dtype=np.float64)
    frame["y_cash"] = 100.0 + 0.1 * np.arange(days, dtype=np.float64)
    return frame


def test_last_value_base_repeats_the_last_observation() -> None:
    series = pd.Series(
        [1.0, 2.0, 5.0], index=pd.date_range("2024-01-01", periods=3, freq="D")
    )
    base = fit_last_value_base(series)
    future = pd.date_range("2024-02-01", periods=4, freq="D")
    assert base.predict(future).tolist() == [5.0, 5.0, 5.0, 5.0]


def test_last_value_base_survives_a_json_round_trip(tmp_path: Path) -> None:
    base = LastValueBase(12.5, pd.Timestamp("2025-03-01"))
    path = tmp_path / "base.json"
    path.write_text(base.to_json(), encoding="utf-8")
    loaded = load_last_value_base(path)
    future = pd.date_range("2025-04-01", periods=3, freq="D")
    assert loaded.predict(future).tolist() == [12.5, 12.5, 12.5]


def test_residual_gap_can_be_overridden() -> None:
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    series = pd.Series(np.arange(400, dtype=np.float64), index=dates)
    with_cfg_gap = oos_residuals(series, fit_last_value_base, CFG)
    without_gap = oos_residuals(series, fit_last_value_base, CFG, gap=0)
    assert with_cfg_gap.iloc[0] == 6.0
    assert without_gap.iloc[0] == 1.0


def test_y_fx_uses_the_random_walk_base_by_default() -> None:
    model = fit_hybrid(_frame(), "y_fx", CFG, residual_fitter=fit_zero)
    assert isinstance(model.base, LastValueBase)
    future = pd.date_range("2030-01-01", periods=2, freq="D")
    assert model.base.predict(future).tolist() == pytest.approx([17.799, 17.799])


def test_y_cash_keeps_the_prophet_base_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mt, "fit_prophet_base", fit_mean_base)
    model = fit_hybrid(_frame(), "y_cash", CFG, residual_fitter=fit_zero)
    assert isinstance(model.base, MeanBase)


def test_random_walk_base_is_saved_and_loaded(tmp_path: Path) -> None:
    model = fit_hybrid(_frame(), "y_fx", CFG, residual_fitter=fit_zero)
    save_model(model, tmp_path)
    info = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert info["base_model"] == "last_value"
    assert "last_value_base.json" in info["artifacts"]
    assert "prophet_model.json" not in info["artifacts"]
    loaded = load_trained_model(tmp_path)
    assert isinstance(loaded.base, LastValueBase)
    future = pd.date_range("2030-01-01", periods=1, freq="D")
    assert loaded.base.predict(future).tolist() == pytest.approx([17.799])
