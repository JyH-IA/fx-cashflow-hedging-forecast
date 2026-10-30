"""Tests for validating the residual tree with exogenous features frozen at the origin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.config import TrainingConfig
from fxhedge.feature_engineering import add_calendar_features
from fxhedge.model_training import (
    BaseForecaster,
    Predictor,
    evaluate_gate,
    fit_hybrid,
    frozen_holdout_features,
)

CFG = TrainingConfig(n_splits=3, gap=5, optuna_trials=1)
SEEN: list[pd.DataFrame] = []


class MeanBase:
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), self.mean, dtype=np.float64)

    def to_json(self) -> str:
        return json.dumps({"mean": self.mean})


class SignalTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return 2.0 * features["x"].to_numpy(dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


class ProbeTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        SEEN.append(features.copy())
        return np.zeros(len(features), dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


def fit_mean_base(series: pd.Series) -> BaseForecaster:
    return MeanBase(float(series.mean()))


def fit_signal(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    return SignalTree(), {}


def fit_probe(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    return ProbeTree(), {}


def _wave(rows: int) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    x = np.sin(2.0 * np.pi * np.arange(rows) / 10.0 + 1.0)
    features = pd.DataFrame({"x": x}, index=dates)
    return features, pd.Series(2.0 * x, index=dates)


def _model_frame(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    frame["level"] = np.arange(days, dtype=np.float64)
    frame["y_fx"] = 17.0 + 0.001 * np.arange(days, dtype=np.float64)
    frame["y_cash"] = 100.0 + 0.1 * np.arange(days, dtype=np.float64)
    return frame


def test_frozen_features_repeat_the_origin_value_in_each_block() -> None:
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    source = add_calendar_features(pd.DataFrame({"date": dates})).set_index("date")
    source["level"] = np.arange(200, dtype=np.float64)
    index = dates[100:190]
    frozen = frozen_holdout_features(source, index, 30)
    assert frozen.index.equals(index)
    assert (frozen["level"].iloc[:30] == 99.0).all()
    assert (frozen["level"].iloc[30:60] == 129.0).all()
    assert (frozen["level"].iloc[60:] == 159.0).all()
    assert frozen["month"].equals(source.loc[index, "month"])


def test_frozen_features_validate_the_block_size() -> None:
    features, _ = _wave(50)
    with pytest.raises(ValueError):
        frozen_holdout_features(features, features.index[10:20], 0)


def test_gate_accepts_a_tree_only_when_true_features_are_used() -> None:
    features, residuals = _wave(400)
    optimistic = evaluate_gate(features, residuals, CFG, fit_signal)
    realistic = evaluate_gate(features, residuals, CFG, fit_signal, carry_source=features)
    assert optimistic.use_tree
    assert not realistic.use_tree
    assert realistic.rmse_hybrid > realistic.rmse_base


def test_hybrid_gate_scores_the_tree_with_frozen_exogenous_features() -> None:
    SEEN.clear()
    frame = _model_frame()
    fit_hybrid(frame, "y_fx", CFG, fit_mean_base, fit_signal, gate_fitter=fit_probe)
    assert SEEN
    holdout = SEEN[0]
    first_block, second_block = holdout.iloc[:90], holdout.iloc[90:]
    assert first_block["level"].nunique() == 1
    assert second_block["level"].nunique() == 1
    assert first_block["level"].iloc[0] != second_block["level"].iloc[0]
    assert first_block["month"].nunique() > 1
