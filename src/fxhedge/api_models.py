"""Response models of the forecast API (kept free of web-framework imports)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

HORIZONS: tuple[int, ...] = (30, 60, 90)
ScenarioName = Literal["base", "alza", "baja"]
DISCLAIMER = (
    "Educational project with synthetic internal data. "
    "Forecasts are illustrative and are not financial advice."
)
_VALUE_COLUMNS = (
    "fx_predicted",
    "cash_expected_usd",
    "fx_lower",
    "fx_upper",
    "cash_lower",
    "cash_upper",
)


class ForecastPoint(BaseModel):
    """One forecast day. Cash values are USD; exchange rates are MXN per USD."""

    model_config = ConfigDict(frozen=True)

    date: dt.date
    fx_predicted: float
    cash_expected_usd: float
    fx_lower: float
    fx_upper: float
    cash_lower: float
    cash_upper: float


class ForecastResponse(BaseModel):
    """Forecast for ``horizon`` days after ``run_date``."""

    model_config = ConfigDict(frozen=True)

    run_date: dt.date
    horizon: int
    scenario: ScenarioName
    disclaimer: str
    points: list[ForecastPoint]


def build_response(
    frame: pd.DataFrame, run_date: dt.date, horizon: int, scenario: ScenarioName
) -> ForecastResponse:
    """Convert an inference table into the API response.

    Raises:
        ValueError: if the horizon is unsupported, a column is missing, a value is not
            finite or the length is wrong.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    missing = [column for column in ("date", *_VALUE_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast table is missing columns: {missing}")
    if len(frame) != horizon:
        raise ValueError(f"Expected {horizon} rows, got {len(frame)}")
    values = frame.loc[:, list(_VALUE_COLUMNS)].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("Forecast contains non-finite values")
    points = [
        ForecastPoint(
            date=pd.Timestamp(row["date"]).date(),
            fx_predicted=float(row["fx_predicted"]),
            cash_expected_usd=float(row["cash_expected_usd"]),
            fx_lower=float(row["fx_lower"]),
            fx_upper=float(row["fx_upper"]),
            cash_lower=float(row["cash_lower"]),
            cash_upper=float(row["cash_upper"]),
        )
        for _, row in frame.iterrows()
    ]
    return ForecastResponse(
        run_date=run_date,
        horizon=horizon,
        scenario=scenario,
        disclaimer=DISCLAIMER,
        points=points,
    )
