"""Model monitoring: PSI drift per variable, rolling error and the retraining decision.

PSI (Population Stability Index) compares the distribution of a variable in a reference
window with a recent window. Common reading: below 0.1 stable, 0.1 to 0.2 watch, above 0.2
significant drift. Thresholds live in ``configs/monitoring.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from fxhedge.config import MonitoringConfig, load_settings, load_yaml_model
from fxhedge.io_utils import read_csv_with_date

logger = logging.getLogger(__name__)

_EPS = 1e-4
_WATCH_FRACTION = 0.5
# Seasonal by construction: comparing them across windows would always show "drift".
CALENDAR_COLUMNS = frozenset(
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


class MonitoringReport(BaseModel):
    """Drift and error summary. ``status`` is green, amber or red per variable."""

    model_config = ConfigDict(frozen=True)

    psi: dict[str, float]
    status: dict[str, str]
    drifted_features: list[str]
    reference_rmse: float | None = None
    current_rmse: float | None = None
    error_ratio: float | None = None


def _proportions(
    reference: NDArray[np.float64], current: NDArray[np.float64], bins: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.unique(reference)
    if len(values) <= bins:
        categories = np.unique(np.concatenate([reference, current]))
        ref_p = np.array([(reference == c).mean() for c in categories])
        cur_p = np.array([(current == c).mean() for c in categories])
        return ref_p, cur_p
    inner = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))[1:-1])
    size = len(inner) + 1
    ref_idx = np.searchsorted(inner, reference, side="right")
    cur_idx = np.searchsorted(inner, current, side="right")
    ref_p = np.bincount(ref_idx, minlength=size) / len(reference)
    cur_p = np.bincount(cur_idx, minlength=size) / len(current)
    return ref_p, cur_p


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of ``current`` against ``reference``.

    Variables with at most ``bins`` distinct reference values are compared category by
    category; others use reference quantile bins.

    Raises:
        ValueError: if ``bins`` is below 2 or either series has no valid values.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ref = reference.dropna().to_numpy(dtype=np.float64)
    cur = current.dropna().to_numpy(dtype=np.float64)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current need at least one valid value")
    ref_p, cur_p = _proportions(ref, cur, bins)
    ref_p = np.clip(ref_p, _EPS, None)
    cur_p = np.clip(cur_p, _EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def rolling_error(y_true: pd.Series, y_pred: pd.Series, window: int = 30) -> pd.DataFrame:
    """Daily error plus rolling RMSE and MAPE (percent) over ``window`` observations.

    Raises:
        ValueError: if the inputs have different lengths or ``window`` is below 2.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if window < 2:
        raise ValueError("window must be at least 2")
    actual = y_true.to_numpy(dtype=np.float64)
    error = actual - y_pred.to_numpy(dtype=np.float64)
    squared = pd.Series(error**2, index=y_true.index)
    pct = pd.Series(np.abs(error) / np.where(actual == 0.0, np.nan, np.abs(actual)) * 100.0)
    pct.index = y_true.index
    return pd.DataFrame(
        {
            "date": y_true.index,
            "error": error,
            "rolling_rmse": np.sqrt(squared.rolling(window).mean()).to_numpy(),
            "rolling_mape": pct.rolling(window).mean().to_numpy(),
        }
    )


def _status(value: float, threshold: float) -> str:
    if value > threshold:
        return "red"
    if value > threshold * _WATCH_FRACTION:
        return "amber"
    return "green"


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(errors.to_numpy(dtype=np.float64) ** 2)))


def build_monitoring_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    errors: pd.DataFrame | None,
    cfg: MonitoringConfig,
) -> MonitoringReport:
    """Compute PSI for every shared numeric, non-calendar column and compare recent error.

    ``errors`` has columns ``date`` and ``error`` (actual minus predicted). Errors dated
    before the first ``current`` date are the reference; the last ``cfg.rolling_window``
    errors are the recent sample.
    """
    shared = [
        column
        for column in reference.columns
        if column in current.columns
        and column != "date"
        and column not in CALENDAR_COLUMNS
        and pd.api.types.is_numeric_dtype(reference[column])
    ]
    psi = {
        column: population_stability_index(reference[column], current[column], cfg.psi_bins)
        for column in shared
    }
    status = {column: _status(value, cfg.psi_threshold) for column, value in psi.items()}
    drifted = sorted(column for column, value in psi.items() if value > cfg.psi_threshold)
    reference_rmse: float | None = None
    current_rmse: float | None = None
    ratio: float | None = None
    if errors is not None and not errors.empty:
        ordered = errors.sort_values("date")
        boundary = current["date"].min()
        past = ordered.loc[ordered["date"] < boundary, "error"]
        recent = ordered["error"].tail(cfg.rolling_window)
        if len(past) > 0 and len(recent) > 0:
            reference_rmse = _rmse(past)
            current_rmse = _rmse(recent)
            ratio = current_rmse / reference_rmse if reference_rmse > 0.0 else None
    return MonitoringReport(
        psi=psi,
        status=status,
        drifted_features=drifted,
        reference_rmse=reference_rmse,
        current_rmse=current_rmse,
        error_ratio=ratio,
    )


def retrain_reasons(report: MonitoringReport, cfg: MonitoringConfig) -> list[str]:
    """Human-readable reasons to retrain; empty when the model looks healthy."""
    reasons: list[str] = []
    if report.drifted_features:
        reasons.append(
            f"PSI above {cfg.psi_threshold} for {len(report.drifted_features)} variable(s): "
            + ", ".join(report.drifted_features[:5])
        )
    if report.error_ratio is not None and report.error_ratio > cfg.error_ratio_threshold:
        reasons.append(
            f"recent error is {report.error_ratio:.2f}x the reference "
            f"(threshold {cfg.error_ratio_threshold})"
        )
    return reasons


def should_retrain(report: MonitoringReport, cfg: MonitoringConfig) -> bool:
    """True when drift or error degradation exceeds the configured thresholds."""
    return bool(retrain_reasons(report, cfg))


def run_monitoring(
    master: pd.DataFrame,
    errors: pd.DataFrame | None,
    cfg: MonitoringConfig,
    current_days: int = 90,
) -> MonitoringReport:
    """Split ``master`` into reference and the last ``current_days`` days and report drift.

    Raises:
        ValueError: if either window ends up empty.
    """
    cutoff = master["date"].max() - pd.Timedelta(days=current_days)
    reference = master[master["date"] <= cutoff]
    current = master[master["date"] > cutoff]
    if reference.empty or current.empty:
        raise ValueError("Not enough history to split reference and current windows")
    return build_monitoring_report(reference, current, errors, cfg)


def write_monitoring_report(report: MonitoringReport, out_dir: Path, run_date: date) -> Path:
    """Write ``<out_dir>/<run_date>.json`` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_date.isoformat()}.json"
    path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return path


def retrain_if_needed(
    report: MonitoringReport, cfg: MonitoringConfig, retrain: Callable[[], int]
) -> bool:
    """Call ``retrain`` only when the report asks for it; return whether it ran."""
    reasons = retrain_reasons(report, cfg)
    if not reasons:
        logger.info("Model looks healthy; no retraining needed")
        return False
    logger.warning("Retraining triggered: %s", "; ".join(reasons))
    retrain()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: write a monitoring report and optionally retrain."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Monitor drift and model error.")
    parser.add_argument(
        "--master", type=Path, default=settings.data_dir / "processed" / "master_features.csv"
    )
    parser.add_argument("--errors", type=Path, default=None, help="CSV with date,error columns")
    parser.add_argument("--current-days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=Path("reports/monitoring"))
    parser.add_argument("--retrain", action="store_true", help="Retrain when thresholds are hit.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_yaml_model(settings.configs_dir / "monitoring.yaml", MonitoringConfig)
    master = read_csv_with_date(args.master)
    errors = read_csv_with_date(args.errors) if args.errors is not None else None
    report = run_monitoring(master, errors, cfg, args.current_days)
    path = write_monitoring_report(report, args.out, date.today())
    logger.info("Wrote %s", path)
    if args.retrain:
        from fxhedge.model_training import main as train_main

        retrain_if_needed(report, cfg, lambda: train_main([]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
