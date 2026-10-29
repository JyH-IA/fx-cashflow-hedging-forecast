"""Tests for the model backtest option and the random-walk note in the hedging report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fxhedge.config import TrainingConfig
from fxhedge.evaluation import (
    BASELINE_MODEL,
    backtest,
    model_forecasters,
    random_walk_forecaster,
    summarize_backtest,
)
from fxhedge.feature_engineering import add_calendar_features
from fxhedge.hedging_backtest import (
    RANDOM_WALK_NOTE,
    HedgeConfig,
    compare_strategies,
    simulate_strategy,
    write_hedging_report,
)
from fxhedge.model_training import BaseForecaster, Predictor

CFG = TrainingConfig(n_splits=2, gap=5, optuna_trials=1)
HEDGE_CFG = HedgeConfig(
    horizon_days=30,
    bank_spread_bps=0.0,
    fixed_fee_usd=0.0,
    rate_mxn_pct=0.0,
    rate_usd_pct=0.0,
)


class MeanBase:
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), self.mean, dtype=np.float64)

    def to_json(self) -> str:
        return "{}"


class ZeroTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


def fit_mean_base(series: pd.Series) -> BaseForecaster:
    return MeanBase(float(series.mean()))


def fit_zero_tree(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    return ZeroTree(), {}


def _model_frame(days: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    frame["cetes_28d"] = 10.0
    frame["y_fx"] = 17.0 + 0.001 * np.arange(days, dtype=np.float64)
    frame["y_cash"] = 100.0 + 0.1 * np.arange(days, dtype=np.float64)
    return frame


def _hedge_data(days: int = 720, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    fx = 17.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, days)))
    cash = 100_000.0 + rng.normal(0.0, 2_000.0, days)
    return pd.DataFrame({"date": dates, "y_fx": fx, "y_cash": cash})


def test_model_forecasters_exposes_hybrid_and_xgboost() -> None:
    forecasters = model_forecasters(_model_frame(), "y_fx", CFG)
    assert set(forecasters) == {"hybrid", "xgboost"}
    assert all(callable(forecaster) for forecaster in forecasters.values())


def test_backtest_reports_a_model_forecaster_next_to_the_baseline() -> None:
    from fxhedge.model_training import make_model_forecaster

    frame = _model_frame()
    models = {
        BASELINE_MODEL: random_walk_forecaster,
        "hybrid": make_model_forecaster(frame, "y_fx", CFG, "hybrid", fit_mean_base, fit_zero_tree),
    }
    summary = summarize_backtest(backtest(frame, "y_fx", [7], CFG, models))
    assert set(summary["model"]) == {BASELINE_MODEL, "hybrid"}
    assert np.isfinite(summary["rmse"].to_numpy()).all()
    assert np.isfinite(summary["rmse_vs_random_walk"].to_numpy()).all()


def test_hedging_report_shows_the_random_walk_note_only_when_given(tmp_path: Path) -> None:
    data = _hedge_data()
    results = {name: simulate_strategy(data, name, HEDGE_CFG) for name in ("none", "fixed")}
    comparison = compare_strategies(results, n_boot=20, block=3, seed=0)
    with_note = tmp_path / "with_note.md"
    without_note = tmp_path / "without_note.md"
    write_hedging_report(comparison, with_note, RANDOM_WALK_NOTE)
    write_hedging_report(comparison, without_note)
    assert RANDOM_WALK_NOTE in with_note.read_text(encoding="utf-8")
    assert "random-walk expectation" not in without_note.read_text(encoding="utf-8")
