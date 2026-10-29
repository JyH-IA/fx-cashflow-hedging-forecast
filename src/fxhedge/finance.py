"""Hedging finance: covered-interest-parity forwards and the value of a partial hedge.

Quotes are MXN per USD. The company receives USD and converts it to MXN, so hedging means
selling USD forward. All figures are illustrative; this is not financial advice.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

DAY_COUNT_BASIS = 360.0
_BPS = 10_000.0


class HedgeResult(BaseModel):
    """Outcome of hedging part of a USD inflow, in MXN unless noted."""

    model_config = ConfigDict(frozen=True)

    hedged_notional_usd: float
    unhedged_notional_usd: float
    mxn_with_hedge: float
    mxn_without_hedge: float
    expected_benefit_mxn: float
    expected_benefit_usd: float
    bank_cost_mxn: float


def forward_rate(spot: float, rate_mxn_pct: float, rate_usd_pct: float, days: int) -> float:
    """Forward USD/MXN from covered interest parity (simple interest, Act/360).

    Raises:
        ValueError: if spot or days are not positive, or a rate is negative.
    """
    if spot <= 0.0 or days <= 0:
        raise ValueError("spot and days must be positive")
    if rate_mxn_pct < 0.0 or rate_usd_pct < 0.0:
        raise ValueError("rates must not be negative")
    growth_mxn = 1.0 + rate_mxn_pct / 100.0 * days / DAY_COUNT_BASIS
    growth_usd = 1.0 + rate_usd_pct / 100.0 * days / DAY_COUNT_BASIS
    return spot * growth_mxn / growth_usd


def forward_points(spot: float, rate_mxn_pct: float, rate_usd_pct: float, days: int) -> float:
    """Forward points: forward rate minus spot, in MXN per USD."""
    return forward_rate(spot, rate_mxn_pct, rate_usd_pct, days) - spot


def hedged_proceeds_mxn(
    notional_usd: float,
    hedge_ratio: float,
    forward: float,
    realized_spot: float,
    bank_spread_bps: float,
    fixed_fee_usd: float,
) -> float:
    """MXN received when ``hedge_ratio`` of the notional is sold forward.

    The bank pays ``forward`` reduced by the spread on the hedged part; the rest is
    converted at ``realized_spot``. The fixed fee (USD) is paid only when hedging.

    Raises:
        ValueError: if an input is outside its valid range.
    """
    if notional_usd < 0.0 or realized_spot <= 0.0 or forward <= 0.0:
        raise ValueError("notional must be >= 0 and rates must be positive")
    if not 0.0 <= hedge_ratio <= 1.0:
        raise ValueError("hedge_ratio must be between 0 and 1")
    if bank_spread_bps < 0.0 or fixed_fee_usd < 0.0:
        raise ValueError("costs must not be negative")
    hedged = notional_usd * hedge_ratio
    effective_forward = forward * (1.0 - bank_spread_bps / _BPS)
    fee_mxn = fixed_fee_usd * realized_spot if hedge_ratio > 0.0 else 0.0
    return hedged * effective_forward + (notional_usd - hedged) * realized_spot - fee_mxn


def compute_hedge_savings(
    notional_usd: float,
    hedge_ratio: float,
    spot_forecast: float,
    forward_rate_value: float,
    bank_spread_bps: float,
    fixed_fee_usd: float,
) -> HedgeResult:
    """Expected MXN benefit of hedging versus converting everything at the forecast spot.

    A negative benefit means the hedge is expected to cost money (the price of certainty).
    """
    with_hedge = hedged_proceeds_mxn(
        notional_usd,
        hedge_ratio,
        forward_rate_value,
        spot_forecast,
        bank_spread_bps,
        fixed_fee_usd,
    )
    without_hedge = notional_usd * spot_forecast
    hedged = notional_usd * hedge_ratio
    spread_cost = hedged * forward_rate_value * bank_spread_bps / _BPS
    fee_cost = fixed_fee_usd * spot_forecast if hedge_ratio > 0.0 else 0.0
    benefit = with_hedge - without_hedge
    return HedgeResult(
        hedged_notional_usd=hedged,
        unhedged_notional_usd=notional_usd - hedged,
        mxn_with_hedge=with_hedge,
        mxn_without_hedge=without_hedge,
        expected_benefit_mxn=benefit,
        expected_benefit_usd=benefit / spot_forecast,
        bank_cost_mxn=spread_cost + fee_cost,
    )
