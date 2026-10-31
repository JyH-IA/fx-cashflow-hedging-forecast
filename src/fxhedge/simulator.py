"""Synthetic internal sales generator with Caribbean-style seasonality.

All values produced here are synthetic. They do not represent any real company.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fxhedge.config import SimulatorConfig
from fxhedge.schemas import SALES_SPEC, validate_frame

_BASE_OCCUPANCY = 0.62
_STREAM_OCCUPANCY = 0
_STREAM_SALES = 1
_STREAM_CASH = 2


def _gaussian_bump(
    day_of_year: NDArray[np.float64], center: float, width: float
) -> NDArray[np.float64]:
    return np.exp(-(((day_of_year - center) / width) ** 2))


def _noise(seed: int, stream: int, scale: float, size: int) -> NDArray[np.float64]:
    """Normal noise from its own random stream.

    Each component has a stream of its own, so the first days of the series do not change
    when ``size`` grows (for example when the end date moves forward).
    """
    rng = np.random.default_rng([seed, stream])
    values: NDArray[np.float64] = rng.normal(0.0, scale, size)
    return values


def seasonal_occupancy(dates: pd.DatetimeIndex, cfg: SimulatorConfig) -> NDArray[np.float64]:
    """Deterministic occupancy signal before noise.

    Components: winter peak (annual cosine), spring break and summer bumps,
    hurricane-season trough (September-October) and a weekend uplift.
    """
    doy = dates.dayofyear.to_numpy(dtype=np.float64)
    winter = np.cos(2.0 * np.pi * (doy - 45.0) / 365.25)
    spring_break = _gaussian_bump(doy, 80.0, 12.0)
    summer = _gaussian_bump(doy, 200.0, 35.0)
    hurricane = _gaussian_bump(doy, 265.0, 30.0)
    weekend = np.isin(dates.dayofweek.to_numpy(), (4, 5)).astype(np.float64)
    amplitude = cfg.seasonality_amplitude
    signal: NDArray[np.float64] = (
        _BASE_OCCUPANCY
        + amplitude * (0.7 * winter + 0.3 * spring_break + 0.9 * summer)
        - cfg.hurricane_penalty * hurricane
        + 0.03 * weekend
    )
    return signal


def simulate_internal_sales(
    cfg: SimulatorConfig, start: date, end: date, seed: int
) -> pd.DataFrame:
    """Generate daily synthetic occupancy, sales and cash-in.

    The noise of each component comes from its own random stream, so extending ``end`` keeps
    the values of the earlier days.

    Returns:
        A DataFrame that satisfies ``SALES_SPEC``.

    Raises:
        ValueError: if ``end`` is before ``start``.
    """
    if end < start:
        raise ValueError("end must not be before start")
    dates = pd.date_range(start=start, end=end, freq="D")
    size = len(dates)

    noise = _noise(seed, _STREAM_OCCUPANCY, cfg.noise_std_pct, size)
    occupancy = np.clip(seasonal_occupancy(dates, cfg) + noise, 0.05, 0.99)

    years = (dates - dates[0]).days.to_numpy(dtype=np.float64) / 365.25
    growth = (1.0 + cfg.annual_growth) ** years
    sales_noise = 1.0 + _noise(seed, _STREAM_SALES, cfg.noise_std_pct, size)
    sales = np.clip(
        cfg.base_daily_sales_usd * (occupancy / _BASE_OCCUPANCY) * growth * sales_noise, 0.0, None
    )
    cash_noise = 1.0 + _noise(seed, _STREAM_CASH, cfg.noise_std_pct / 2.0, size)
    cash_in = np.clip(sales * cfg.cash_conversion_ratio * cash_noise, 0.0, None)

    frame = pd.DataFrame(
        {
            "date": dates,
            "occupancy_rate": occupancy.astype(np.float64),
            "sales_usd": sales.astype(np.float64),
            "cash_in_usd": cash_in.astype(np.float64),
        }
    )
    return validate_frame(frame, SALES_SPEC)
