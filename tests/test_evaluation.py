"""Tests for evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.config import TrainingConfig
from fxhedge.evaluation import (
    BASELINE_MODEL,
    backtest,
    baseline_forecasts,
    compute_metrics,
    default_forecasters,
    fold_origins,
    interval_coverage,
    make_seasonal_naive,
    random_walk_forecaster,
    sarima_forecast,
    summarize_backtest,
    write_report,
)


def _periodic_frame(days: int = 500, period: int = 7) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    pattern = np.arange(1.0, period + 1.0)
    values = np.resize(pattern, days) + 10.0
    return pd.DataFrame({"date": dates, "y_fx": values})


def test_compute_metrics_known_values() -> None:
    report = compute_metrics(pd.Series([100.0, 200.0]), pd.Series([110.0, 180.0]))
    assert report.mape == pytest.approx(10.0)
    assert report.rmse == pytest.approx(np.sqrt((100.0 + 400.0) / 2.0))
    assert report.mae == pytest.approx(15.0)
    assert report.n == 2


def test_compute_metrics_skips_zero_actuals_for_mape() -> None:
    report = compute_metrics(pd.Series([0.0, 100.0]), pd.Series([5.0, 110.0]))
    assert report.mape == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([1.0, 2.0], [1.0]), ([], [])],
    ids=["length-mismatch", "empty"],
)
def test_compute_metrics_rejects_bad_inputs(actual: list[float], predicted: list[float]) -> None:
    with pytest.raises(ValueError):
        compute_metrics(pd.Series(actual, dtype=float), pd.Series(predicted, dtype=float))


def test_interval_coverage() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    coverage = interval_coverage(y, y - 0.5, pd.Series([1.2, 2.2, 2.9, 4.5]))
    assert coverage == pytest.approx(0.75)


def test_random_walk_repeats_last_value_with_future_dates() -> None:
    frame = _periodic_frame(20)
    train = frame.set_index("date")["y_fx"]
    path = random_walk_forecaster(train, 3)
    assert path.tolist() == [float(train.iloc[-1])] * 3
    assert path.index[0] == train.index[-1] + pd.Timedelta(days=1)


def test_seasonal_naive_reproduces_periodic_series() -> None:
    frame = _periodic_frame(60)
    train = frame.set_index("date")["y_fx"].iloc[:50]
    expected = frame.set_index("date")["y_fx"].iloc[50:60]
    path = make_seasonal_naive(7)(train, 10)
    assert np.allclose(path.to_numpy(), expected.to_numpy())


def test_seasonal_naive_needs_enough_history() -> None:
    short = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError):
        make_seasonal_naive(7)(short, 3)


def test_fold_origins_spacing_and_error() -> None:
    origins = fold_origins(n_rows=1000, n_splits=5, max_horizon=90, min_train=365)
    assert origins[0] == 365
    assert origins[-1] == 910
    assert origins == sorted(origins)
    with pytest.raises(ValueError):
        fold_origins(n_rows=400, n_splits=5, max_horizon=90, min_train=365)


def test_baseline_forecasts_columns_and_dates() -> None:
    frame = _periodic_frame(100)
    out = baseline_forecasts(frame, "y_fx", 5)
    assert list(out.columns) == ["date", BASELINE_MODEL, "seasonal_naive_7"]
    assert len(out) == 5
    assert out["date"].iloc[0] == frame["date"].iloc[-1] + pd.Timedelta(days=1)


def test_baseline_forecasts_rejects_unknown_target() -> None:
    with pytest.raises(ValueError):
        baseline_forecasts(_periodic_frame(50), "missing", 5)


def test_backtest_seasonal_naive_beats_random_walk_on_periodic_data() -> None:
    frame = _periodic_frame(600)
    cfg = TrainingConfig(n_splits=3)
    results = backtest(frame, "y_fx", [7, 14], cfg)
    assert set(results["model"]) == set(default_forecasters("y_fx"))
    assert len(results) == 3 * 2 * 2
    summary = summarize_backtest(results)
    seasonal = summary[summary["model"] == "seasonal_naive_7"]
    assert (seasonal["rmse"] < 1e-9).all()
    assert (seasonal["rmse_vs_random_walk"] < 1.0).all()
    walk = summary[summary["model"] == BASELINE_MODEL]
    assert np.allclose(walk["rmse_vs_random_walk"], 1.0)


def test_backtest_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        backtest(_periodic_frame(200), "y_fx", [30], TrainingConfig())


def test_write_report_creates_json_and_markdown(tmp_path: Path) -> None:
    frame = _periodic_frame(600)
    summary = summarize_backtest(backtest(frame, "y_fx", [7], TrainingConfig(n_splits=2)))
    write_report(summary, tmp_path / "reports")
    records = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert {record["model"] for record in records} == set(default_forecasters("y_fx"))
    markdown = (tmp_path / "reports" / "backtest.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Backtest results")


def test_sarima_forecast_returns_future_path() -> None:
    pytest.importorskip("statsmodels")
    series = _periodic_frame(120).set_index("date")["y_fx"]
    path = sarima_forecast(series, 5, (1, 0, 0), (0, 0, 0, 0))
    assert len(path) == 5
    assert path.index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert np.isfinite(path.to_numpy()).all()
