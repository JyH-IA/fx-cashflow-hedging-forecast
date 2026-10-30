"""Tests for the horizon-sized blocks used to build the hybrid's residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fxhedge.config import TrainingConfig
from fxhedge.model_training import BaseForecaster, oos_residuals, residual_split_count


class MeanBase:
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), self.mean, dtype=np.float64)

    def to_json(self) -> str:
        return "{}"


def fit_mean_base(series: pd.Series) -> BaseForecaster:
    return MeanBase(float(series.mean()))


def _series(days: int) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    return pd.Series(np.arange(days, dtype=np.float64), index=dates)


def test_split_count_respects_gap_minimum_window_and_cap() -> None:
    cfg = TrainingConfig()
    assert residual_split_count(1095, cfg) == 7
    assert residual_split_count(1000, cfg) == 6
    assert residual_split_count(300, cfg) == 0
    assert residual_split_count(10_000, cfg) == cfg.residual_max_splits


def test_long_series_uses_horizon_sized_blocks() -> None:
    cfg = TrainingConfig(n_splits=5, gap=90, optuna_trials=1)
    series = _series(1500)
    residuals = oos_residuals(series, fit_mean_base, cfg)
    blocks = residual_split_count(1500, cfg)
    assert blocks >= cfg.residual_min_splits
    assert len(residuals) == blocks * cfg.residual_test_size
    assert residuals.index.min() == series.index[len(series) - len(residuals)]
    assert residuals.index.is_unique


def test_short_series_keeps_the_previous_split() -> None:
    cfg = TrainingConfig(n_splits=3, gap=5, optuna_trials=1)
    residuals = oos_residuals(_series(400), fit_mean_base, cfg)
    assert len(residuals) == 3 * (400 // (cfg.n_splits + 1))


def test_residual_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(residual_test_size=0)
    with pytest.raises(ValidationError):
        TrainingConfig(residual_min_splits=1)
