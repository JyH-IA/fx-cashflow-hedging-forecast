"""Feature engineering: series alignment, lags, rolling windows and calendar flags.

Leakage rules: lagged features use only past values, rolling statistics are computed on
the series shifted by one day, and same-day target-related columns are dropped from the
final table.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import holidays
import pandas as pd

from fxhedge.config import Settings, TrainingConfig, load_settings, load_yaml_model
from fxhedge.io_utils import read_csv_with_date, write_csv
from fxhedge.schemas import MACRO_SPEC, MASTER_SPEC, SALES_SPEC, validate_frame

logger = logging.getLogger(__name__)

EXOG_COLUMNS: tuple[str, ...] = ("cetes_28d", "fed_funds", "inpc", "cancun_arrivals")
HIGH_SEASON_MONTHS: frozenset[int] = frozenset({12, 1, 2, 3, 4, 7, 8})
HURRICANE_MONTHS: frozenset[int] = frozenset(range(6, 12))
_LEAKY_COLUMNS: tuple[str, ...] = (
    "usdmxn_fix",
    "usdmxn_yf",
    "sales_usd",
    "cash_in_usd",
    "occupancy_rate",
)


def align_series(macro: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Inner-join macro and sales tables by ``date`` and build the targets.

    Non-banking days are filled forward. ``y_fx`` is the Banxico FIX rate, falling back
    to the Yahoo Finance close; ``y_cash`` is the daily cash-in.

    Raises:
        DataContractError: if an input table violates its contract.
    """
    validate_frame(macro, MACRO_SPEC)
    validate_frame(sales, SALES_SPEC)
    merged = macro.merge(sales, on="date", how="inner").sort_values("date").reset_index(drop=True)
    value_columns = [column for column in merged.columns if column != "date"]
    merged[value_columns] = merged[value_columns].ffill()
    merged["y_fx"] = merged["usdmxn_fix"].fillna(merged["usdmxn_yf"])
    merged["y_cash"] = merged["cash_in_usd"]
    return merged


def add_lags_and_rollings(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    windows: Sequence[int],
) -> pd.DataFrame:
    """Add lag, rolling mean and rolling std features for ``columns``.

    Rolling statistics use the series shifted by one day, so the value of day ``t``
    never enters its own feature.

    Raises:
        ValueError: if a column is missing, a lag is not positive or a window is below 2.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    if any(lag < 1 for lag in lags):
        raise ValueError("Lags must be positive integers")
    if any(window < 2 for window in windows):
        raise ValueError("Windows must be at least 2")
    new_columns: dict[str, pd.Series] = {}
    for column in columns:
        series = df[column]
        for lag in lags:
            new_columns[f"{column}_lag{lag}"] = series.shift(lag)
        shifted = series.shift(1)
        for window in windows:
            rolling = shifted.rolling(window=window)
            new_columns[f"{column}_roll{window}_mean"] = rolling.mean()
            new_columns[f"{column}_roll{window}_std"] = rolling.std()
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekday, month, quarter, weekend, Mexican holiday and season flags."""
    out = df.copy()
    dates = out["date"]
    years = range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    holiday_dates = set(holidays.country_holidays("MX", years=years).keys())
    out["day_of_week"] = dates.dt.dayofweek.astype(int)
    out["month"] = dates.dt.month.astype(int)
    out["quarter"] = dates.dt.quarter.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday_mx"] = dates.dt.date.isin(holiday_dates).astype(int)
    out["is_high_season"] = out["month"].isin(HIGH_SEASON_MONTHS).astype(int)
    out["is_hurricane_season"] = out["month"].isin(HURRICANE_MONTHS).astype(int)
    return out


def build_master_features(
    macro_path: Path,
    sales_path: Path,
    out_path: Path,
    lags: Sequence[int] = (7, 14, 30),
    windows: Sequence[int] = (7, 14, 30),
) -> pd.DataFrame:
    """Build and save ``master_features.csv`` from the raw macro and sales files.

    Exogenous columns that are entirely empty (for example a missing optional CSV)
    are dropped with a warning.

    Raises:
        ValueError: if no rows remain after the lag and rolling warm-up.
    """
    aligned = align_series(read_csv_with_date(macro_path), read_csv_with_date(sales_path))
    present_exog = [column for column in EXOG_COLUMNS if aligned[column].notna().any()]
    for column in EXOG_COLUMNS:
        if column not in present_exog:
            logger.warning("Dropping empty exogenous column: %s", column)
    lag_sources = ["y_fx", "y_cash", "occupancy_rate", *present_exog]
    featured = add_lags_and_rollings(aligned, lag_sources, lags, windows)
    featured = add_calendar_features(featured)
    drop_columns = [*_LEAKY_COLUMNS, *(c for c in EXOG_COLUMNS if c not in present_exog)]
    master = featured.drop(columns=drop_columns).dropna().reset_index(drop=True)
    if master.empty:
        raise ValueError("No rows left after feature construction; extend the date range")
    validate_frame(master, MASTER_SPEC)
    write_csv(master, out_path)
    return master


def _default_paths(settings: Settings) -> tuple[Path, Path, Path]:
    raw = settings.data_dir / "raw"
    return (
        raw / "macro_data.csv",
        raw / "internal_sales_simulated.csv",
        settings.data_dir / "processed" / "master_features.csv",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    settings = load_settings()
    macro_default, sales_default, out_default = _default_paths(settings)
    parser = argparse.ArgumentParser(description="Build master_features.csv.")
    parser.add_argument("--macro", type=Path, default=macro_default)
    parser.add_argument("--sales", type=Path, default=sales_default)
    parser.add_argument("--out", type=Path, default=out_default)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_yaml_model(settings.configs_dir / "training.yaml", TrainingConfig)
    master = build_master_features(args.macro, args.sales, args.out, cfg.lags, cfg.windows)
    logger.info("Wrote %d rows and %d columns to %s", len(master), master.shape[1], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
