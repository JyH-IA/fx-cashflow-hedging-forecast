"""Pure helpers behind the Streamlit dashboard (no UI imports, so they are easy to test)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from fxhedge.config import BusinessConfig
from fxhedge.finance import compute_hedge_savings, forward_rate, hedged_proceeds_mxn
from fxhedge.model_training import LastValueBase, Target, TrainedModel, ZeroResidual

DEFAULT_RATIOS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
HEDGE_LEVELS: tuple[tuple[str, float], ...] = (
    ("None (0%)", 0.0),
    ("Low (25%)", 0.25),
    ("Balanced (50%)", 0.5),
    ("High (75%)", 0.75),
    ("Full (100%)", 1.0),
)
HEDGE_LABEL_BY_RATIO: dict[float, str] = {ratio: label for label, ratio in HEDGE_LEVELS}
HORIZON_LABELS: dict[int, str] = {
    30: "1 month (30 days)",
    60: "2 months (60 days)",
    90: "3 months (90 days)",
}
HORIZON_PERIODS: dict[int, str] = {30: "month", 60: "2 months", 90: "3 months"}
TARGET_LABELS: dict[str, str] = {"y_fx": "USD/MXN", "y_cash": "USD cash-in"}
DATA_SOURCE_LABELS: dict[str, str] = {
    "synthetic": "SYNTHETIC demo data, not real market data",
    "banxico": "Banxico market data (internal sales are synthetic)",
    "unknown": "not recorded, run make sample or make ingest again",
}
METRICS_FILE = Path("reports") / "metrics.json"
MODEL_LABELS: dict[str, str] = {
    "hybrid": "Hybrid",
    "xgboost": "XGBoost",
    "sarima": "SARIMA",
    "random_walk": "Random walk",
    "seasonal_naive_7": "Seasonal naive (7 days)",
    "seasonal_naive_365": "Seasonal naive (365 days)",
}
_REQUIRED = ("fx_predicted", "fx_lower", "fx_upper", "cash_expected_usd")
_COMPARISON_COLUMNS = ("target", "model", "horizon", "rmse_vs_random_walk")
_BASELINE_PREFIXES = ("random_walk", "seasonal_naive")


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


def notional_step(expected_cash_usd: float) -> float:
    """Slider step for the amount to protect: about 1 % of the expected inflow, at least 1.

    It is always a float, because Streamlit requires the minimum, maximum, value and step of
    a slider to share the same numeric type.
    """
    return float(max(round(expected_cash_usd / 100.0), 1))


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


def format_compact_money(value: float, currency: str = "MXN") -> str:
    """Format an amount with a K, M or B suffix, for example ``$9.6M MXN``."""
    sign = "-" if value < 0 else ""
    amount = abs(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if amount >= threshold * 0.99995:
            return f"{sign}${amount / threshold:,.1f}{suffix} {currency}"
    return f"{sign}${amount:,.0f} {currency}"


def escape_markdown_dollars(text: str) -> str:
    """Escape dollar signs so Streamlit does not read a pair of them as a formula."""
    return text.replace("$", "\\$")


def data_source_line(source: str, as_of: date, spot: float) -> str:
    """One line with the data source, the date of the last observation and the last rate."""
    label = DATA_SOURCE_LABELS.get(source, DATA_SOURCE_LABELS["unknown"])
    return f"Data as of {as_of:%Y-%m-%d} | Last USD/MXN: {spot:.4f} | Source: {label}"


def hedge_ratio(label: str) -> float:
    """Hedge ratio behind a label of ``HEDGE_LEVELS``.

    Raises:
        ValueError: if the label is not one of the hedge levels.
    """
    ratios = dict(HEDGE_LEVELS)
    if label not in ratios:
        raise ValueError(f"Unknown hedge level: {label}")
    return ratios[label]


def summary_kpis(table: pd.DataFrame, selected_ratio: float) -> dict[str, float]:
    """Key figures that compare the selected hedge ratio with no hedge.

    ``protected`` is how much the bad-case outcome (the low end of the interval) rises, and
    ``narrowing_pct`` is how much the gap between the bad and the good case shrinks.

    Raises:
        ValueError: if the table lacks the no-hedge row or the selected ratio.
    """
    base = table.loc[table["hedge_ratio"] == 0.0]
    chosen = table.loc[table["hedge_ratio"] == selected_ratio]
    if base.empty or chosen.empty:
        raise ValueError("The impact table must include the no-hedge row and the selected ratio")
    none_row = base.iloc[0]
    chosen_row = chosen.iloc[0]
    range_none = float(none_row["range_mxn"])
    range_selected = float(chosen_row["range_mxn"])
    narrowing = (1.0 - range_selected / range_none) * 100.0 if range_none > 0.0 else 0.0
    return {
        "worst_none": float(none_row["worst_case_mxn"]),
        "worst_selected": float(chosen_row["worst_case_mxn"]),
        "protected": float(chosen_row["worst_case_mxn"]) - float(none_row["worst_case_mxn"]),
        "range_none": range_none,
        "range_selected": range_selected,
        "narrowing_pct": narrowing,
        "bank_cost": float(chosen_row["bank_cost_mxn"]),
    }


def summary_headline(
    kpis: Mapping[str, float], horizon: int, ratio: float, notional_usd: float
) -> str:
    """One sentence that answers the business question for the selected hedge level."""
    period = HORIZON_PERIODS.get(horizon, f"{horizon} days")
    amount = format_compact_money(notional_usd, "USD")
    if ratio <= 0.0:
        return (
            f"Without a hedge, a bad but plausible outcome for the next {period} of USD income "
            f"({amount}) is {format_compact_money(kpis['worst_none'])}. "
            "Pick a hedge level to see what it protects."
        )
    return (
        f"Hedging {ratio:.0%} of the next {period} of USD income ({amount}) lifts the bad-case "
        f"outcome by {format_compact_money(kpis['protected'])} and narrows the possible swing "
        f"by {kpis['narrowing_pct']:.0f}%, at a bank cost of "
        f"{format_compact_money(kpis['bank_cost'])}."
    )


def friendly_impact_table(table: pd.DataFrame) -> pd.DataFrame:
    """Hedge levels side by side in plain words and compact amounts."""
    labels = [
        HEDGE_LABEL_BY_RATIO.get(float(ratio), f"{ratio:.0%}") for ratio in table["hedge_ratio"]
    ]
    shown = pd.DataFrame({"Hedge level": labels})
    shown["Forward rate"] = table["forward_rate"].map("{:.4f}".format)
    columns = (
        ("expected_proceeds_mxn", "Expected income"),
        ("worst_case_mxn", "Bad case"),
        ("best_case_mxn", "Good case"),
        ("bank_cost_mxn", "Bank cost"),
    )
    for column, label in columns:
        shown[label] = table[column].map(format_compact_money)
    return shown


def base_only_targets(models: Mapping[Target, TrainedModel]) -> list[str]:
    """Targets whose residual tree was rejected by the validation gate."""
    return [str(target) for target, model in models.items() if isinstance(model.tree, ZeroResidual)]


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


def load_model_comparison(path: Path = METRICS_FILE) -> pd.DataFrame | None:
    """Read the backtest summary that ``fxhedge.evaluation`` writes.

    Returns ``None`` when the file does not exist.

    Raises:
        ValueError: if the file is not valid JSON, is not a list of records or lacks the
            columns needed for the comparison.
    """
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of records")
    frame = pd.DataFrame(records)
    missing = [column for column in _COMPARISON_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def comparison_table(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    """RMSE relative to the random walk, with models as rows and horizons as columns.

    Models are sorted from best (lowest average ratio) to worst. Values are text with two
    decimals, and ``n/a`` where the ratio is missing. The result is empty when ``metrics``
    has no rows for ``target``.
    """
    subset = metrics.loc[metrics["target"] == target]
    if subset.empty:
        return pd.DataFrame()
    ratios = pd.to_numeric(subset["rmse_vs_random_walk"], errors="coerce")
    wide = (
        subset.assign(ratio=ratios)
        .pivot_table(
            index="model", columns="horizon", values="ratio", aggfunc="mean", dropna=False
        )
        .sort_index(axis=1)
    )
    wide = wide.loc[wide.mean(axis=1).sort_values().index]
    table = pd.DataFrame(index=[MODEL_LABELS.get(str(name), str(name)) for name in wide.index])
    for horizon in wide.columns:
        table[f"{horizon} days"] = [
            "n/a" if pd.isna(value) else f"{value:.2f}" for value in wide[horizon]
        ]
    return table


def reliability_verdicts(metrics: pd.DataFrame) -> dict[str, str]:
    """One plain sentence per target on whether a trained model beats the simple baselines.

    The best trained model (hybrid, XGBoost or SARIMA, by average ratio) is compared with the
    best simple baseline (the random walk or a seasonal naive). A gain below 3 % is reported
    as no meaningful gain, below 15 % as modest and anything above as clear. Targets without
    both kinds of model are skipped.
    """
    verdicts: dict[str, str] = {}
    for target, label in TARGET_LABELS.items():
        subset = metrics.loc[metrics["target"] == target]
        if subset.empty:
            continue
        ratios = pd.to_numeric(subset["rmse_vs_random_walk"], errors="coerce")
        averages = subset.assign(ratio=ratios).groupby("model")["ratio"].mean().dropna()
        means = {str(name): float(value) for name, value in averages.items()}
        baselines = {n: v for n, v in means.items() if n.startswith(_BASELINE_PREFIXES)}
        trained = {n: v for n, v in means.items() if n not in baselines}
        if not baselines or not trained:
            continue
        best_name = min(trained, key=lambda name: trained[name])
        base_name = min(baselines, key=lambda name: baselines[name])
        gain = 1.0 - trained[best_name] / baselines[base_name]
        best = MODEL_LABELS.get(best_name, best_name)
        base = MODEL_LABELS.get(base_name, base_name)
        if gain < 0.03:
            text = f"no trained model clearly beats the best simple baseline ({base})"
        elif gain < 0.15:
            text = (
                f"{best} is modestly better than the best simple baseline ({base}): "
                f"about {gain:.0%} less error"
            )
        else:
            text = (
                f"{best} is clearly better than the best simple baseline ({base}): "
                f"about {gain:.0%} less error"
            )
        verdicts[target] = f"{label}: {text}."
    return verdicts
