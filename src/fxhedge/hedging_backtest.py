"""Historical simulation of hedging strategies on monthly USD inflows.

Assumptions: the USD inflow of the next ``horizon_days`` is known when the hedge is placed
(the study isolates FX risk), MXN costs are budgeted at the origin spot, and the forward
is priced with covered interest parity. This is a simulation, not an A/B test.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fxhedge.config import BusinessConfig, load_settings, load_yaml_model
from fxhedge.finance import forward_rate, hedged_proceeds_mxn
from fxhedge.io_utils import read_csv_with_date

logger = logging.getLogger(__name__)

Strategy = Literal["none", "fixed", "forecast"]
STRATEGIES: tuple[Strategy, ...] = ("none", "fixed", "forecast")
BASELINE_STRATEGY: Strategy = "none"
REQUIRED_COLUMNS = ("date", "y_fx", "y_cash")
RANDOM_WALK_NOTE = (
    "**Warning:** no model forecasts were provided, so the `forecast` strategy uses the "
    "random-walk expectation (spot at the origin). With a higher MXN than USD rate the "
    "forward is always above the spot, so this strategy always hedges the high ratio and "
    "behaves like a fixed hedge. Its results do not measure forecast skill."
)


def _days(count: int) -> pd.Timedelta:
    return pd.Timedelta(count, unit="D")


class HedgeConfig(BaseModel):
    """Parameters of the simulation. Ratios are fractions of the USD inflow."""

    model_config = ConfigDict(frozen=True)

    horizon_days: int = Field(30, ge=1)
    fixed_ratio: float = Field(0.5, ge=0.0, le=1.0)
    forecast_low_ratio: float = Field(0.25, ge=0.0, le=1.0)
    forecast_high_ratio: float = Field(0.75, ge=0.0, le=1.0)
    mxn_cost_share: float = Field(0.82, ge=0.0, le=1.0)
    bank_spread_bps: float = Field(15.0, ge=0.0)
    fixed_fee_usd: float = Field(25.0, ge=0.0)
    rate_mxn_pct: float = Field(10.0, ge=0.0)
    rate_usd_pct: float = Field(4.5, ge=0.0)

    @classmethod
    def from_business(cls, business: BusinessConfig, horizon_days: int = 30) -> HedgeConfig:
        """Build a simulation config from the business assumptions."""
        return cls(
            horizon_days=horizon_days,
            fixed_ratio=business.fixed_hedge_ratio,
            mxn_cost_share=business.mxn_cost_share,
            bank_spread_bps=business.bank_spread_bps,
            fixed_fee_usd=business.fixed_fee_usd,
            rate_mxn_pct=business.rate_mxn_pct,
            rate_usd_pct=business.rate_usd_pct,
        )


@dataclass(frozen=True)
class StrategyComparison:
    """Summary table plus the bootstrap settings used to build it."""

    summary: pd.DataFrame
    n_boot: int
    block: int


def _month_origins(dates: pd.Series, horizon_days: int) -> list[pd.Timestamp]:
    first_per_month = dates.groupby([dates.dt.year, dates.dt.month]).min()
    last_allowed = dates.max() - _days(horizon_days)
    return [pd.Timestamp(day) for day in first_per_month if day <= last_allowed]


def _ratio(
    strategy: Strategy,
    cfg: HedgeConfig,
    forward: float,
    forecast: float | None,
) -> float:
    if strategy == "none":
        return 0.0
    if strategy == "fixed":
        return cfg.fixed_ratio
    if forecast is None:
        raise ValueError("The forecast strategy needs a forecast for every origin")
    return cfg.forecast_high_ratio if forecast < forward else cfg.forecast_low_ratio


def simulate_strategy(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: HedgeConfig,
    fx_forecast: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate one strategy with a hedge placed on the first day of each month.

    ``fx_forecast`` is indexed by origin date and holds the expected spot at settlement.
    The forecast strategy hedges more when the forecast is below the forward (the forward
    locks in a better rate than expected) and less otherwise.

    Raises:
        ValueError: if required columns are missing or no origin fits in the data.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = df.sort_values("date").set_index("date")
    origins = _month_origins(data.index.to_series(), cfg.horizon_days)
    if not origins:
        raise ValueError("Not enough data for a single hedging origin")
    rows: list[dict[str, object]] = []
    for origin in origins:
        settle = origin + _days(cfg.horizon_days)
        window = data.loc[(data.index > origin) & (data.index <= settle), "y_cash"]
        exposure = float(window.sum())
        spot0 = float(data.loc[:origin, "y_fx"].iloc[-1])
        settle_spot = float(data.loc[:settle, "y_fx"].iloc[-1])
        forward = forward_rate(spot0, cfg.rate_mxn_pct, cfg.rate_usd_pct, cfg.horizon_days)
        forecast = None
        if strategy == "forecast":
            if fx_forecast is None or origin not in fx_forecast.index:
                raise ValueError(f"Missing forecast for origin {origin.date()}")
            forecast = float(fx_forecast.loc[origin])
        ratio = _ratio(strategy, cfg, forward, forecast)
        proceeds = hedged_proceeds_mxn(
            exposure, ratio, forward, settle_spot, cfg.bank_spread_bps, cfg.fixed_fee_usd
        )
        unhedged = exposure * settle_spot
        costs = cfg.mxn_cost_share * exposure * spot0
        rows.append(
            {
                "origin": origin,
                "settle_date": settle,
                "exposure_usd": exposure,
                "spot0": spot0,
                "forward": forward,
                "settle_spot": settle_spot,
                "hedge_ratio": ratio,
                "proceeds_mxn": proceeds,
                "costs_mxn": costs,
                "margin_mxn": proceeds - costs,
                "hedge_cost_mxn": unhedged - proceeds if ratio > 0.0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _block_indices(size: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(size / block))
    starts = rng.integers(0, max(size - block + 1, 1), size=n_blocks)
    indices = np.concatenate([np.arange(start, start + block) for start in starts])
    return indices[:size]


def compare_strategies(
    results: Mapping[str, pd.DataFrame], n_boot: int, block: int, seed: int
) -> StrategyComparison:
    """Compare strategies against ``none`` with a moving-block bootstrap.

    Reports mean, standard deviation and worst month of the MXN margin, total hedge cost,
    the reduction of margin variance and the mean margin difference, both with 95 % intervals.

    Raises:
        ValueError: if ``none`` is missing or strategies have different origins.
    """
    if BASELINE_STRATEGY not in results:
        raise ValueError("results must include the 'none' strategy")
    if n_boot < 1 or block < 1:
        raise ValueError("n_boot and block must be positive")
    base = results[BASELINE_STRATEGY]["margin_mxn"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = [_block_indices(len(base), min(block, len(base)), rng) for _ in range(n_boot)]
    rows: list[dict[str, object]] = []
    for name, table in results.items():
        if not table["origin"].equals(results[BASELINE_STRATEGY]["origin"]):
            raise ValueError(f"Strategy {name} has different origins than 'none'")
        margin = table["margin_mxn"].to_numpy(dtype=np.float64)
        reduction = [1.0 - np.var(margin[i]) / np.var(base[i]) for i in samples]
        difference = [float(np.mean(margin[i] - base[i])) for i in samples]
        rows.append(
            {
                "strategy": name,
                "mean_margin_mxn": float(margin.mean()),
                "std_margin_mxn": float(margin.std()),
                "worst_margin_mxn": float(margin.min()),
                "total_hedge_cost_mxn": float(table["hedge_cost_mxn"].sum()),
                "var_reduction": float(1.0 - np.var(margin) / np.var(base)),
                "var_reduction_lo": float(np.percentile(reduction, 2.5)),
                "var_reduction_hi": float(np.percentile(reduction, 97.5)),
                "mean_diff_mxn": float(np.mean(margin - base)),
                "mean_diff_lo": float(np.percentile(difference, 2.5)),
                "mean_diff_hi": float(np.percentile(difference, 97.5)),
            }
        )
    return StrategyComparison(summary=pd.DataFrame(rows), n_boot=n_boot, block=block)


def write_hedging_report(
    comparison: StrategyComparison, out_path: Path, forecast_note: str | None = None
) -> None:
    """Write the comparison as a Markdown report.

    ``forecast_note`` is added under the title when the forecast strategy did not use
    real model forecasts.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = list(comparison.summary.columns)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for record in comparison.summary.itertuples(index=False):
        cells = [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in record]
        lines.append("| " + " | ".join(cells) + " |")
    notes = [
        "# Historical hedging simulation",
        "",
        "Illustrative simulation on synthetic cash flows and real or sample FX data.",
        "It is not an A/B test and not financial advice.",
        "",
    ]
    if forecast_note is not None:
        notes.extend([forecast_note, ""])
    notes.extend(
        [
            *lines,
            "",
            f"Intervals: {comparison.n_boot} moving-block bootstrap resamples, "
            f"block length {comparison.block} months (95 % percentile interval).",
            "`var_reduction` is the fall in margin variance versus no hedge; "
            "`mean_diff_mxn` is the average monthly margin change versus no hedge.",
        ]
    )
    out_path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: simulate the three strategies and write the report."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Simulate hedging strategies.")
    parser.add_argument(
        "--master", type=Path, default=settings.data_dir / "processed" / "master_features.csv"
    )
    parser.add_argument("--forecasts", type=Path, default=None, help="CSV: date,fx_forecast")
    parser.add_argument("--out", type=Path, default=Path("reports/hedging_backtest.md"))
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--block", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    business = load_yaml_model(settings.configs_dir / "business.yaml", BusinessConfig)
    cfg = HedgeConfig.from_business(business, args.horizon)
    master = read_csv_with_date(args.master)
    used_random_walk = args.forecasts is None
    if args.forecasts is not None:
        table = read_csv_with_date(args.forecasts)
        forecasts = table.set_index("date")["fx_forecast"]
    else:
        logger.warning("No forecasts file: using the random-walk expectation (spot at origin)")
        forecasts = None
    results: dict[str, pd.DataFrame] = {}
    for strategy in STRATEGIES:
        if strategy == "forecast" and forecasts is None:
            origins = simulate_strategy(master, "none", cfg)["origin"]
            spot = master.set_index("date")["y_fx"]
            expected = [float(spot.loc[:day].iloc[-1]) for day in origins]
            forecasts = pd.Series(expected, index=origins)
        results[strategy] = simulate_strategy(master, strategy, cfg, forecasts)
    comparison = compare_strategies(results, args.n_boot, args.block, settings.seed)
    write_hedging_report(comparison, args.out, RANDOM_WALK_NOTE if used_random_walk else None)
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
