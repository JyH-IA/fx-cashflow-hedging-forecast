"""Fully synthetic macro and sales data, so the pipeline runs without any API token.

Everything produced here is invented: the exchange-rate series is a random walk, not market
data. Use it to exercise the pipeline, never to draw conclusions about the real market.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fxhedge.config import SimulatorConfig, load_settings, load_yaml_model
from fxhedge.data_source import write_data_source
from fxhedge.io_utils import write_csv
from fxhedge.schemas import MACRO_SPEC, validate_frame
from fxhedge.simulator import simulate_internal_sales

logger = logging.getLogger(__name__)

_START_FX = 18.0
_DAILY_FX_VOLATILITY = 0.005
_FX_NOISE_YAHOO = 0.01


def simulate_macro(start: date, end: date, seed: int) -> pd.DataFrame:
    """Synthetic daily macro table that satisfies ``MACRO_SPEC``.

    Exchange rates are missing on weekends, like real banking series. Inflation and Cancun
    arrivals are monthly values repeated within the month.

    Raises:
        ValueError: if ``end`` is before ``start``.
    """
    if end < start:
        raise ValueError("end must not be before start")
    dates = pd.date_range(start=start, end=end, freq="D")
    size = len(dates)
    rng = np.random.default_rng(seed)

    fix = _START_FX * np.exp(np.cumsum(rng.normal(0.0, _DAILY_FX_VOLATILITY, size)))
    yahoo = fix + rng.normal(0.0, _FX_NOISE_YAHOO, size)

    cetes = np.empty(size)
    level = 10.0
    for index in range(size):
        level += 0.01 * (10.0 - level) + rng.normal(0.0, 0.02)
        cetes[index] = level
    fed_funds = 4.5 + 0.5 * np.sin(2.0 * np.pi * np.arange(size) / 1460.0)

    month_index = (dates.year - dates[0].year) * 12 + (dates.month - dates[0].month)
    inpc = 130.0 * (1.003 ** month_index.to_numpy(dtype=np.float64))
    seasonal = 1.0 + 0.2 * np.cos(2.0 * np.pi * (dates.month.to_numpy() - 2) / 12.0)
    arrivals = 1_500_000.0 * seasonal

    frame = pd.DataFrame(
        {
            "date": dates,
            "usdmxn_fix": fix,
            "usdmxn_yf": yahoo,
            "cetes_28d": cetes,
            "fed_funds": fed_funds,
            "inpc": inpc,
            "cancun_arrivals": arrivals,
        }
    )
    weekend = dates.dayofweek >= 5
    frame.loc[weekend, ["usdmxn_fix", "usdmxn_yf"]] = np.nan
    return validate_frame(frame, MACRO_SPEC)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: write synthetic macro and sales CSVs to ``data/raw``."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Generate fully synthetic sample data.")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "raw")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    sim_cfg = load_yaml_model(settings.configs_dir / "simulator.yaml", SimulatorConfig)
    start = args.start or sim_cfg.start
    end = args.end or sim_cfg.end
    write_csv(simulate_macro(start, end, settings.seed), args.out_dir / "macro_data.csv")
    sales = simulate_internal_sales(sim_cfg, start, end, settings.seed)
    write_csv(sales, args.out_dir / "internal_sales_simulated.csv")
    write_data_source(args.out_dir, "synthetic")
    logger.warning("Wrote SYNTHETIC data to %s (not real market data)", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
