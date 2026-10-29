"""Model evaluation: metrics, baselines, SARIMA and rolling-origin backtesting.

Every forecaster maps a training series and a horizon to a forecast path. The metric for
horizon ``h`` is computed over the first ``h`` days of that path, which matches how the
30/60/90-day forecasts are used for hedging decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fxhedge.config import TrainingConfig, load_settings, load_yaml_model
from fxhedge.io_utils import read_csv_with_date

logger = logging.getLogger(__name__)

Forecaster = Callable[[pd.Series, int], pd.Series]

SEASONAL_PERIODS: dict[str, int] = {"y_fx": 7, "y_cash": 365}
BASELINE_MODEL = "random_walk"
MIN_TRAIN_ROWS = 365
_METRIC_COLUMNS = ("mape", "rmse", "mae")


class MetricsReport(BaseModel):
    """Error metrics of a forecast. ``mape`` is a percentage and may be NaN."""

    model_config = ConfigDict(frozen=True)

    mape: float
    rmse: float
    mae: float
    n: int


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> MetricsReport:
    """Compute MAPE (%), RMSE and MAE.

    Positions where ``y_true`` is zero are skipped for MAPE only.

    Raises:
        ValueError: if the inputs are empty or have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty inputs")
    actual = y_true.to_numpy(dtype=np.float64)
    predicted = y_pred.to_numpy(dtype=np.float64)
    errors = actual - predicted
    nonzero = actual != 0.0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100.0)
        if bool(nonzero.any())
        else float("nan")
    )
    return MetricsReport(
        mape=mape,
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        n=len(actual),
    )


def interval_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Return the fraction of observations that fall inside ``[lower, upper]``."""
    if not (len(y_true) == len(lower) == len(upper)) or len(y_true) == 0:
        raise ValueError("Inputs must be non-empty and have the same length")
    inside = (y_true.to_numpy() >= lower.to_numpy()) & (y_true.to_numpy() <= upper.to_numpy())
    return float(np.mean(inside))


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    last = pd.Timestamp(train.index[-1])
    return pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")


def random_walk_forecaster(train: pd.Series, horizon: int) -> pd.Series:
    """Repeat the last observed value for every future day."""
    return pd.Series(float(train.iloc[-1]), index=_future_index(train, horizon))


def make_seasonal_naive(period: int) -> Forecaster:
    """Build a forecaster that repeats the last full seasonal cycle of length ``period``."""
    if period < 1:
        raise ValueError("period must be positive")

    def forecaster(train: pd.Series, horizon: int) -> pd.Series:
        if len(train) < period:
            raise ValueError(f"Need at least {period} observations, got {len(train)}")
        cycle = train.iloc[-period:].to_numpy(dtype=np.float64)
        values = np.resize(cycle, horizon)
        return pd.Series(values, index=_future_index(train, horizon))

    return forecaster


def sarima_forecast(
    series: pd.Series,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model with statsmodels and forecast ``horizon`` days ahead.

    ``series`` must have a daily DatetimeIndex.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    daily = series.astype(float).asfreq("D")
    fitted = SARIMAX(daily, order=order, seasonal_order=seasonal_order).fit(disp=False)
    forecast = fitted.get_forecast(steps=horizon).predicted_mean
    forecast.index = _future_index(daily, horizon)
    result: pd.Series = forecast.astype(float)
    return result


def make_sarima_forecaster(
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    max_train_rows: int = 730,
) -> Forecaster:
    """Build a SARIMA forecaster that fits on the most recent ``max_train_rows`` days."""

    def forecaster(train: pd.Series, horizon: int) -> pd.Series:
        return sarima_forecast(train.iloc[-max_train_rows:], horizon, order, seasonal_order)

    return forecaster


def default_forecasters(target: str) -> dict[str, Forecaster]:
    """Baselines used for every target: random walk and seasonal naive."""
    period = SEASONAL_PERIODS.get(target, 7)
    return {
        BASELINE_MODEL: random_walk_forecaster,
        f"seasonal_naive_{period}": make_seasonal_naive(period),
    }


def baseline_forecasts(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Forecast ``horizon`` days after the last row with each baseline.

    Returns:
        Columns ``date`` plus one column per baseline.
    """
    series = _target_series(df, target)
    paths = {name: fc(series, horizon) for name, fc in default_forecasters(target).items()}
    frame = pd.DataFrame(paths)
    frame.insert(0, "date", frame.index)
    return frame.reset_index(drop=True)


def _target_series(df: pd.DataFrame, target: str) -> pd.Series:
    if target not in df.columns or "date" not in df.columns:
        raise ValueError(f"DataFrame must contain 'date' and '{target}'")
    return df.set_index("date")[target].astype(float).sort_index()


def fold_origins(n_rows: int, n_splits: int, max_horizon: int, min_train: int) -> list[int]:
    """Return evenly spaced training sizes for an expanding-window backtest.

    Raises:
        ValueError: if the series is too short for the requested folds.
    """
    last_end = n_rows - max_horizon
    if last_end < min_train:
        raise ValueError(
            f"Series too short: {n_rows} rows, need at least {min_train + max_horizon}"
        )
    points = np.linspace(min_train, last_end, n_splits).round().astype(int)
    return sorted({int(point) for point in points})


def backtest(
    df: pd.DataFrame,
    target: str,
    horizons: Sequence[int],
    cfg: TrainingConfig,
    forecasters: Mapping[str, Forecaster] | None = None,
    min_train: int = MIN_TRAIN_ROWS,
) -> pd.DataFrame:
    """Run an expanding-window, rolling-origin backtest.

    ``cfg.n_splits`` sets the number of origins. ``cfg.gap`` applies to feature-based
    models trained in ``model_training``; path forecasters here never see data after
    their origin, so no gap is needed.

    Returns:
        One row per fold, model and horizon with ``mape``, ``rmse`` and ``mae``.
    """
    series = _target_series(df, target)
    models = dict(forecasters) if forecasters is not None else default_forecasters(target)
    max_horizon = max(horizons)
    origins = fold_origins(len(series), cfg.n_splits, max_horizon, min_train)
    rows: list[dict[str, object]] = []
    for fold, end in enumerate(origins):
        train = series.iloc[:end]
        actual = series.iloc[end : end + max_horizon]
        for name, forecaster in models.items():
            path = forecaster(train, max_horizon)
            for horizon in horizons:
                report = compute_metrics(actual.iloc[:horizon], path.iloc[:horizon])
                rows.append(
                    {
                        "target": target,
                        "model": name,
                        "horizon": horizon,
                        "fold": fold,
                        "mape": report.mape,
                        "rmse": report.rmse,
                        "mae": report.mae,
                    }
                )
    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Average fold metrics by target, model and horizon.

    Adds ``rmse_vs_random_walk``: values below 1 mean the model beats the random walk.
    """
    keys = ["target", "model", "horizon"]
    summary = results.groupby(keys, as_index=False)[list(_METRIC_COLUMNS)].mean()
    baseline = summary.loc[summary["model"] == BASELINE_MODEL, ["target", "horizon", "rmse"]]
    baseline = baseline.rename(columns={"rmse": "rmse_random_walk"})
    merged = summary.merge(baseline, on=["target", "horizon"], how="left")
    merged["rmse_vs_random_walk"] = merged["rmse"] / merged["rmse_random_walk"]
    return merged.drop(columns=["rmse_random_walk"])


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, divider]
    for record in frame.itertuples(index=False):
        cells = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in record]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    """Write ``metrics.json`` and ``backtest.md`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = summary.astype(object).where(summary.notna(), None).to_dict(orient="records")
    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    note = (
        "A model is only useful if `rmse_vs_random_walk` is below 1. "
        "For exchange rates a random walk is often hard to beat; report the result as it is.\n"
    )
    body = "# Backtest results\n\n" + _markdown_table(summary) + "\n\n" + note
    (out_dir / "backtest.md").write_text(body, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: backtest baselines (optionally SARIMA) and write reports."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Backtest baselines and write reports.")
    parser.add_argument(
        "--master", type=Path, default=settings.data_dir / "processed" / "master_features.csv"
    )
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--with-sarima", action="store_true", help="Also backtest SARIMA (slow).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_yaml_model(settings.configs_dir / "training.yaml", TrainingConfig)
    master = read_csv_with_date(args.master)
    frames: list[pd.DataFrame] = []
    for target in ("y_fx", "y_cash"):
        models = default_forecasters(target)
        if args.with_sarima:
            models["sarima"] = make_sarima_forecaster()
        frames.append(backtest(master, target, cfg.horizons, cfg, models))
    summary = summarize_backtest(pd.concat(frames, ignore_index=True))
    write_report(summary, args.out)
    logger.info("Wrote reports to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
