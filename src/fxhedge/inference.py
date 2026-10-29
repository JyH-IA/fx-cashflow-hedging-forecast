"""Inference: 30/60/90-day forecasts of USD/MXN and cash-in with prediction intervals.

Future exogenous values are carried forward from the last known row. Scenarios shock every
non-calendar feature by a fixed percentage. Intervals are NOT value-at-risk: they are
empirical prediction intervals built from the historical distribution of h-day log changes
(a conservative proxy) unless error samples from a backtest are supplied.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fxhedge.config import Settings, load_settings
from fxhedge.io_utils import read_csv_with_date
from fxhedge.model_training import (
    TARGETS,
    Target,
    TrainedModel,
    future_feature_frame,
    load_trained_model,
    predict_values,
)

logger = logging.getLogger(__name__)

Scenario = Literal["base", "alza", "baja"]
HORIZONS: tuple[int, ...] = (30, 60, 90)
SCENARIO_SHOCKS: Mapping[str, float] = {"base": 0.0, "alza": 0.10, "baja": -0.10}
_CALENDAR_COLUMNS = frozenset(
    {
        "day_of_week",
        "month",
        "quarter",
        "is_weekend",
        "is_holiday_mx",
        "is_high_season",
        "is_hurricane_season",
    }
)
OUTPUT_COLUMNS = (
    "date",
    "fx_predicted",
    "cash_expected_usd",
    "fx_lower",
    "fx_upper",
    "cash_lower",
    "cash_upper",
)


def validate_request(horizon: int, scenario: str) -> None:
    """Raise ``ValueError`` unless the horizon and scenario are supported."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    if scenario not in SCENARIO_SHOCKS:
        raise ValueError(f"scenario must be one of {tuple(SCENARIO_SHOCKS)}, got {scenario}")


def apply_scenario(features: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Shock every non-calendar column by the scenario percentage (base leaves it as is)."""
    shock = SCENARIO_SHOCKS[scenario]
    out = features.copy()
    for column in out.columns:
        if column not in _CALENDAR_COLUMNS:
            out[column] = out[column] * (1.0 + shock)
    return out


def log_change_samples(
    series: pd.Series, horizon: int, window_days: int = 1095
) -> NDArray[np.float64]:
    """Median-centered h-day log changes of the most recent ``window_days`` of history.

    Raises:
        ValueError: if the series is too short or contains non-positive values.
    """
    recent = series.dropna().tail(window_days + horizon).astype(float)
    if len(recent) <= horizon + 10 or bool((recent <= 0.0).any()):
        raise ValueError("Need more positive history to estimate prediction intervals")
    changes = np.log(recent.shift(-horizon) / recent).dropna().to_numpy(dtype=np.float64)
    return changes - float(np.median(changes))


def interval_bounds(
    point: NDArray[np.float64], log_errors: NDArray[np.float64], level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Multiplicative bounds ``point * exp(quantile)`` at the central ``level``.

    Raises:
        ValueError: if ``level`` is outside (0, 1), errors are empty or a point is not positive.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if len(log_errors) == 0 or bool((point <= 0.0).any()):
        raise ValueError("Need error samples and positive point forecasts")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(log_errors, [tail, 1.0 - tail])
    return point * float(np.exp(low)), point * float(np.exp(high))


def predict_forecast(
    models: Mapping[Target, TrainedModel],
    history: pd.DataFrame,
    run_date: date,
    horizon: int,
    scenario: Scenario = "base",
    level: float = 0.95,
    error_samples: Mapping[Target, NDArray[np.float64]] | None = None,
) -> pd.DataFrame:
    """Forecast the ``horizon`` days after ``run_date``.

    Returns:
        Columns ``date``, ``fx_predicted``, ``cash_expected_usd`` and the interval bounds.

    Raises:
        ValueError: for unsupported inputs or when there is no history up to ``run_date``.
    """
    validate_request(horizon, scenario)
    known = history[history["date"] <= pd.Timestamp(run_date)]
    if known.empty:
        raise ValueError("No history available up to run_date")
    dates = pd.date_range(pd.Timestamp(run_date) + pd.Timedelta(days=1), periods=horizon, freq="D")
    columns: dict[str, NDArray[np.float64]] = {}
    specs: list[tuple[Target, str, str, str]] = [
        ("y_fx", "fx_predicted", "fx_lower", "fx_upper"),
        ("y_cash", "cash_expected_usd", "cash_lower", "cash_upper"),
    ]
    for target, point_name, low_name, high_name in specs:
        model = models[target]
        future = future_feature_frame(known, dates, model.feature_columns)
        features = apply_scenario(future, scenario)
        point = np.clip(predict_values(model, features), 1e-9, None)
        errors = (
            error_samples[target]
            if error_samples is not None and target in error_samples
            else log_change_samples(known.set_index("date")[target], horizon)
        )
        lower, upper = interval_bounds(point, errors, level)
        columns[point_name] = point
        columns[low_name] = lower
        columns[high_name] = upper
    frame = pd.DataFrame({"date": dates, **columns})
    return frame.loc[:, list(OUTPUT_COLUMNS)]


def load_models(models_dir: Path) -> dict[Target, TrainedModel]:
    """Load the hybrid model of every target, verifying artifact hashes."""
    return {target: load_trained_model(models_dir / target) for target in TARGETS}


def predict(
    run_date: date,
    horizon: int,
    scenario: Scenario = "base",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load models and history from disk and forecast ``horizon`` days after ``run_date``."""
    validate_request(horizon, scenario)
    active = settings or load_settings()
    history = read_csv_with_date(active.data_dir / "processed" / "master_features.csv")
    return predict_forecast(load_models(active.models_dir), history, run_date, horizon, scenario)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: print the forecast as JSON records."""
    parser = argparse.ArgumentParser(description="Forecast USD/MXN and cash-in.")
    parser.add_argument("--run-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=30)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_SHOCKS), default="base")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = predict(args.run_date, args.horizon, args.scenario)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    print(result.to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
