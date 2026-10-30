"""Tests for the validation gate that can disable the hybrid's residual tree."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fxhedge.config import TrainingConfig
from fxhedge.feature_engineering import add_calendar_features
from fxhedge.model_training import (
    BASE_ARTIFACT,
    LIGHTGBM_ARTIFACT,
    METADATA_FILE,
    BaseForecaster,
    Predictor,
    ZeroResidual,
    evaluate_gate,
    fit_hybrid,
    future_feature_frame,
    load_trained_model,
    predict_values,
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


class ExactTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return 2.0 * features["x"].to_numpy(dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


class ConstantTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 5.0, dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


def fit_mean_base(series: pd.Series) -> BaseForecaster:
    return MeanBase(float(series.mean()))


def load_mean_base(path: Path) -> BaseForecaster:
    return MeanBase(float(json.loads(path.read_text(encoding="utf-8"))["mean"]))


def fit_exact(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    return ExactTree(), {}


def fit_constant(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    return ConstantTree(), {}


def _signal(rows: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    x = rng.normal(size=rows)
    features = pd.DataFrame({"x": x}, index=dates)
    residuals = pd.Series(2.0 * x + rng.normal(0.0, 0.1, rows), index=dates)
    return features, residuals


def _model_frame(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    frame["cetes_28d"] = 10.0
    frame["y_fx"] = 17.0 + 0.001 * np.arange(days, dtype=np.float64)
    frame["y_cash"] = 100.0 + 0.1 * np.arange(days, dtype=np.float64)
    return frame


def test_gate_keeps_a_tree_that_explains_the_residuals() -> None:
    features, residuals = _signal(400)
    result = evaluate_gate(features, residuals, CFG, fit_exact)
    assert result.use_tree
    assert result.rmse_hybrid < result.rmse_base
    assert result.holdout_rows == 100


def test_gate_rejects_a_tree_that_makes_the_error_worse() -> None:
    features, residuals = _signal(400)
    result = evaluate_gate(features, residuals, CFG, fit_constant)
    assert not result.use_tree
    assert result.rmse_hybrid > result.rmse_base


def test_gate_requires_the_configured_improvement() -> None:
    features, residuals = _signal(400)
    strict = TrainingConfig(n_splits=3, gap=5, optuna_trials=1, gate_min_improvement=0.99)
    assert not evaluate_gate(features, residuals, strict, fit_exact).use_tree


def test_gate_rejects_the_tree_when_there_are_too_few_rows_to_validate() -> None:
    features, residuals = _signal(40)
    result = evaluate_gate(features, residuals, CFG, fit_exact)
    assert not result.use_tree
    assert math.isnan(result.rmse_base)
    assert math.isnan(result.rmse_hybrid)


def test_hybrid_without_a_gate_always_fits_the_tree() -> None:
    model = fit_hybrid(_model_frame(), "y_fx", CFG, fit_mean_base, fit_constant)
    assert model.gate is None
    assert isinstance(model.tree, ConstantTree)


def test_rejected_tree_is_not_saved_and_the_model_reloads(tmp_path: Path) -> None:
    frame = _model_frame()
    model = fit_hybrid(
        frame, "y_fx", CFG, fit_mean_base, fit_constant, gate_fitter=fit_constant
    )
    assert isinstance(model.tree, ZeroResidual)
    assert model.gate is not None
    assert not model.gate.use_tree

    out_dir = tmp_path / "y_fx"
    save_model(model, out_dir)
    info = json.loads((out_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert info["residual_model"] == "none"
    assert info["gate"]["use_tree"] is False
    assert set(info["artifacts"]) == {BASE_ARTIFACT}
    assert not (out_dir / LIGHTGBM_ARTIFACT).exists()

    loaded = load_trained_model(out_dir, load_mean_base)
    assert isinstance(loaded.tree, ZeroResidual)
    dates = pd.date_range("2024-03-01", periods=5, freq="D")
    features = future_feature_frame(frame, dates, model.feature_columns)
    assert np.allclose(predict_values(model, features), predict_values(loaded, features))
