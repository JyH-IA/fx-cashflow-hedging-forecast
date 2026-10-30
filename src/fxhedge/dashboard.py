"""Pure helpers behind the Streamlit dashboard (no UI imports, so they are easy to test)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from fxhedge.config import BusinessConfig
from fxhedge.finance import compute_hedge_savings, forward_rate, hedged_proceeds_mxn
from fxhedge.model_training import LastValueBase, Target, TrainedModel, ZeroResidual

DEFAULT_RATIOS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_LABELS: dict[str, str] = {"y_fx": "USD/MXN", "y_cash": "USD cash-in"}
_REQUIRED = ("fx_predicted", "fx_lower", "fx_upper", "cash_expected_usd")


def _check_forecast(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Forecast table is empty")


def expected_cash_in_usd(frame: pd.DataFrame) -> float:
    """Total expected USD inflow over the forecast horizon."""
    _check_forecast(frame)
    return float(frame["cash_expected_usd"].sum())


def forecast_summary(frame: pd.DataFrame, spot: float) -> dict[str, float]:
    """Headline figures: settlement forecast, its interval and the change versus spot."""
    _check_forecast(frame)
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    last = frame.iloc[-1]
    settlement = float(last["fx_predicted"])
    return {
        "spot": spot,
        "settlement_fx": settlement,
        "settlement_lower": float(last["fx_lower"]),
        "settlement_upper": float(last["fx_upper"]),
        "change_pct": (settlement / spot - 1.0) * 100.0,
        "expected_cash_usd": expected_cash_in_usd(frame),
    }


def impact_table(
    frame: pd.DataFrame,
    spot: float,
    notional_usd: float,
    ratios: Sequence[float],
    business: BusinessConfig,
    horizon: int,
) -> pd.DataFrame:
    """MXN outcome of several hedge ratios: expected, worst and best case in the interval.

    The forward is priced from ``spot`` with covered interest parity. Worst and best case
    use the lower and upper interval bounds of the settlement-day forecast.

    Raises:
        ValueError: if inputs are invalid or the forecast table is incomplete.
    """
    _check_forecast(frame)
    if notional_usd < 0.0:
        raise ValueError("notional_usd must not be negative")
    last = frame.iloc[-1]
    forward = forward_rate(spot, business.rate_mxn_pct, business.rate_usd_pct, horizon)
    rows: list[dict[str, float]] = []
    for ratio in ratios:
        expected = compute_hedge_savings(
            notional_usd,
            ratio,
            float(last["fx_predicted"]),
            forward,
            business.bank_spread_bps,
            business.fixed_fee_usd,
        )
        at_bounds = [
            hedged_proceeds_mxn(
                notional_usd,
                ratio,
                forward,
                float(last[bound]),
                business.bank_spread_bps,
                business.fixed_fee_usd,
            )
            for bound in ("fx_lower", "fx_upper")
        ]
        rows.append(
            {
                "hedge_ratio": ratio,
                "forward_rate": forward,
                "expected_proceeds_mxn": expected.mxn_with_hedge,
                "worst_case_mxn": min(at_bounds),
                "best_case_mxn": max(at_bounds),
                "range_mxn": max(at_bounds) - min(at_bounds),
                "expected_benefit_mxn": expected.expected_benefit_mxn,
                "bank_cost_mxn": expected.bank_cost_mxn,
            }
        )
    return pd.DataFrame(rows)


def commission_saving_mxn(
    notional_usd: float,
    selected_ratio: float,
    spot: float,
    business: BusinessConfig,
    horizon: int,
) -> float:
    """Bank cost avoided by hedging ``selected_ratio`` instead of the whole notional."""
    forward = forward_rate(spot, business.rate_mxn_pct, business.rate_usd_pct, horizon)

    def cost(ratio: float) -> float:
        return compute_hedge_savings(
            notional_usd, ratio, spot, forward, business.bank_spread_bps, business.fixed_fee_usd
        ).bank_cost_mxn

    return cost(1.0) - cost(selected_ratio)


def format_mxn(value: float) -> str:
    """Format an amount as ``$1,234,567 MXN``."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f} MXN"


def base_only_notices(models: Mapping[Target, TrainedModel]) -> list[str]:
    """One notice for each target whose residual tree was rejected by the validation gate.

    Such a forecast comes from the base model alone. The scenarios shock only the features of
    the residual tree, so the selector does not change it. With a random-walk base the line
    is flat at the last observed value.
    """
    notices: list[str] = []
    for target, model in models.items():
        if not isinstance(model.tree, ZeroResidual):
            continue
        label = TARGET_LABELS.get(target, target)
        if isinstance(model.base, LastValueBase):
            detail = "it repeats the last observed value, so the line is flat"
        else:
            detail = "it follows trend and seasonality only"
        notices.append(
            f"{label}: the validation gate rejected the residual model, so the forecast uses "
            f"the base model alone ({detail}). The scenario selector does not change it."
        )
    return notices
