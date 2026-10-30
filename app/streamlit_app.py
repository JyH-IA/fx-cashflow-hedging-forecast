"""Executive dashboard for the FX hedging forecast (Streamlit). Run with ``make app``.

The first screen answers one question: what does a hedge protect? Details sit in later tabs.
All data shown is synthetic or illustrative. This is not financial advice.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fxhedge.api_models import DISCLAIMER, HORIZONS
from fxhedge.config import BusinessConfig, load_settings, load_yaml_model
from fxhedge.dashboard import (
    DEFAULT_RATIOS,
    HEDGE_LABEL_BY_RATIO,
    HEDGE_LEVELS,
    HORIZON_LABELS,
    METRICS_FILE,
    TARGET_LABELS,
    base_only_notices,
    base_only_targets,
    comparison_table,
    forecast_summary,
    format_compact_money,
    friendly_impact_table,
    hedge_ratio,
    impact_table,
    load_model_comparison,
    notional_step,
    reliability_verdicts,
    summary_headline,
    summary_kpis,
)
from fxhedge.inference import Scenario, load_models, predict_forecast
from fxhedge.io_utils import read_csv_with_date
from fxhedge.model_training import ModelIntegrityError

SCENARIOS = ("base", "alza", "baja")
DEFAULT_LEVEL = "Balanced (50%)"


@st.cache_data(ttl=3600, show_spinner="Loading forecast...")
def load_forecast(horizon: int, scenario: str, run_date: date) -> tuple[pd.DataFrame, float]:
    """Forecast table and the last observed spot at ``run_date``."""
    settings = load_settings()
    history = read_csv_with_date(settings.data_dir / "processed" / "master_features.csv")
    frame = predict_forecast(
        load_models(settings.models_dir), history, run_date, horizon, cast(Scenario, scenario)
    )
    known = history.loc[history["date"] <= pd.Timestamp(run_date), "y_fx"]
    return frame, float(known.iloc[-1])


@st.cache_data(ttl=3600)
def load_notices() -> list[str]:
    """Notices for the targets whose forecast comes from the base model alone."""
    return base_only_notices(load_models(load_settings().models_dir))


@st.cache_data(ttl=3600)
def load_base_only_targets() -> list[str]:
    """Targets whose forecast comes from the base model alone."""
    return base_only_targets(load_models(load_settings().models_dir))


def forecast_chart(frame: pd.DataFrame) -> go.Figure:
    """Line chart of the USD/MXN forecast with its prediction interval."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=frame["date"], y=frame["fx_upper"], line={"width": 0}, showlegend=False)
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["fx_lower"],
            fill="tonexty",
            line={"width": 0},
            name="Plausible range (95%)",
        )
    )
    figure.add_trace(go.Scatter(x=frame["date"], y=frame["fx_predicted"], name="Expected rate"))
    figure.update_layout(
        title="USD/MXN: expected rate and plausible range",
        yaxis_title="MXN per USD",
        xaxis_title="Date",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def outcome_range_chart(table: pd.DataFrame, selected_ratio: float) -> go.Figure:
    """Bars from the bad case to the good case for each hedge level; the range narrows."""
    labels = [HEDGE_LABEL_BY_RATIO.get(float(ratio), f"{ratio:.0%}") for ratio in table["hedge_ratio"]]
    colors = ["#1f77b4" if ratio == selected_ratio else "#c7d7ea" for ratio in table["hedge_ratio"]]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=labels,
            x=table["range_mxn"],
            base=table["worst_case_mxn"],
            orientation="h",
            marker_color=colors,
            name="Possible outcomes",
            customdata=table[["worst_case_mxn", "best_case_mxn"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>Bad case: $%{customdata[0]:,.0f} MXN"
                "<br>Good case: $%{customdata[1]:,.0f} MXN<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["expected_proceeds_mxn"],
            y=labels,
            mode="markers",
            marker={"symbol": "diamond", "size": 10, "color": "#222222"},
            name="Expected income",
        )
    )
    figure.update_layout(
        title="What we could receive in MXN, from a bad case to a good case",
        xaxis={"title": "MXN received", "tickprefix": "$", "tickformat": ".3s"},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def read_verdicts() -> dict[str, str]:
    """One plain sentence per target on how much the models beat simple baselines."""
    try:
        metrics = load_model_comparison()
    except ValueError:
        return {}
    return {} if metrics is None else reliability_verdicts(metrics)


def render_reliability() -> None:
    """Backtest comparison of the models, read from ``reports/metrics.json``."""
    st.subheader("Which forecasting method works best?")
    try:
        metrics = load_model_comparison()
    except ValueError as exc:
        st.warning(f"The backtest summary could not be read: {exc}")
        return
    if metrics is None:
        st.info(
            f"No backtest summary found at {METRICS_FILE}. Run `make evaluate-full` to create "
            "it with the hybrid, XGBoost and SARIMA (`make evaluate` adds only the baselines)."
        )
        return
    for text in reliability_verdicts(metrics).values():
        st.markdown(f"- {text}")
    updated = date.fromtimestamp(METRICS_FILE.stat().st_mtime)
    st.caption(
        "How to read the tables: each number is a model's average error divided by the error of "
        "simply repeating the last known value (the random walk). Below 1.00 is better than "
        "that shortcut and above 1.00 is worse. "
        f"Source: {METRICS_FILE}, updated {updated:%Y-%m-%d}."
    )
    for target in ("y_fx", "y_cash"):
        table = comparison_table(metrics, target)
        if table.empty:
            continue
        st.markdown(f"**{TARGET_LABELS[target]}**")
        st.dataframe(table)


def render_technical(horizon: int) -> None:
    """How the forecasts are built, what the gate decided, a glossary and the limits."""
    st.subheader("How the forecasts are built")
    st.markdown(
        "- Each forecast starts from a base model: Prophet (trend and seasonality) for USD "
        "cash-in and the last known rate for USD/MXN.\n"
        "- A second model (LightGBM) may correct the base, but only if it improves the error on "
        "recent data. This check is called the validation gate.\n"
        f"- The plausible range is the 95% historical range of {horizon}-day moves. It is not "
        "a guarantee and not a value-at-risk figure."
    )
    notices = load_notices()
    if notices:
        st.markdown("**What the validation gate decided**")
        for notice in notices:
            st.info(notice)
    with st.expander("Glossary"):
        st.markdown(
            "- **Hedge / forward:** an agreement to sell USD at a fixed rate on a future date.\n"
            "- **Bad case / good case:** the low and high ends of the 95% plausible range.\n"
            "- **Random walk:** repeating the last known value, the usual benchmark for "
            "exchange rates.\n"
            "- **Validation gate:** the check that decides whether the second model is used.\n"
            "- **RMSE:** the typical size of the forecast error."
        )
    st.caption(
        "Assumptions: forward priced with covered interest parity; "
        "bank spread and fixed fee from configs/business.yaml; "
        "exogenous variables held at their last known value."
    )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="FX hedging forecast", layout="wide")
    st.title("Protecting our USD income from peso swings")
    st.caption(DISCLAIMER)

    st.sidebar.header("Your question")
    horizon = int(
        st.sidebar.radio(
            "Planning horizon",
            HORIZONS,
            format_func=lambda days: HORIZON_LABELS.get(days, f"{days} days"),
        )
    )
    level = str(
        st.sidebar.select_slider(
            "How much to hedge",
            options=[label for label, _ in HEDGE_LEVELS],
            value=DEFAULT_LEVEL,
        )
    )
    ratio = hedge_ratio(level)
    with st.sidebar.expander("Advanced settings"):
        scenario = str(st.selectbox("Scenario", SCENARIOS, index=0))
        run_date = st.date_input("Run date", value=date.today())

    try:
        frame, spot = load_forecast(horizon, scenario, run_date)
    except (FileNotFoundError, ModelIntegrityError, ValueError) as exc:
        st.error(
            "The forecast is not available. Train the models first "
            "(make ingest features train) and check the processed data."
        )
        st.caption(f"Reason: {type(exc).__name__}")
        st.stop()

    business = load_yaml_model(load_settings().configs_dir / "business.yaml", BusinessConfig)
    summary = forecast_summary(frame, spot)
    expected = summary["expected_cash_usd"]
    notional = float(
        st.sidebar.slider(
            "USD to protect",
            min_value=0.0,
            max_value=max(expected * 2.0, 1.0),
            value=float(expected),
            step=notional_step(expected),
            format="%.0f",
            help="Starts at the USD income we expect over the planning horizon.",
        )
    )
    table = impact_table(frame, spot, notional, DEFAULT_RATIOS, business, horizon)
    kpis = summary_kpis(table, ratio)

    summary_tab, hedging_tab, reliability_tab, technical_tab = st.tabs(
        ["Summary", "Hedging", "How reliable is it?", "Technical details"]
    )

    with summary_tab:
        st.subheader(summary_headline(kpis, horizon, ratio, notional))
        first, second, third, fourth = st.columns([2, 1, 1, 1])
        first.metric(
            "Downside protected in a bad case",
            format_compact_money(kpis["protected"]),
            help="How much the bad-case outcome improves compared with not hedging.",
        )
        second.metric(
            "Swing in possible outcomes",
            f"{kpis['narrowing_pct']:.0f}% narrower",
            help="How much the gap between the bad and the good case shrinks.",
        )
        third.metric("Bank cost of this hedge", format_compact_money(kpis["bank_cost"]))
        fourth.metric(
            "Expected USD income",
            format_compact_money(summary["expected_cash_usd"], "USD"),
        )
        st.caption(
            f"Bad case is the low end of the 95% historical range of {horizon}-day moves in "
            "USD/MXN. It is a plausible downside, not a guarantee."
        )
        st.plotly_chart(outcome_range_chart(table, ratio))
        st.markdown("**How much to trust these numbers**")
        verdicts = read_verdicts()
        if verdicts:
            for text in verdicts.values():
                st.markdown(f"- {text}")
            st.caption("The details are in the tab 'How reliable is it?'.")
        else:
            st.caption("Run `make evaluate-full` to see how reliable the forecasts are.")

    with hedging_tab:
        st.subheader("Hedge levels side by side")
        st.dataframe(friendly_impact_table(table), hide_index=True)
        st.caption(
            "Income in MXN for the amount to protect. Bad and good case are the ends of the 95% "
            "plausible range. The forward rate sits above today's rate because of the interest "
            "rates assumed in configs/business.yaml (Mexico above the US): part of any apparent "
            "gain from hedging is that gap, not a prediction."
        )
        st.plotly_chart(forecast_chart(frame))
        if "y_fx" in load_base_only_targets():
            st.caption(
                "The models have no reliable view on the direction of USD/MXN, so the line "
                "starts from the last known rate and the band shows the plausible range."
            )

    with reliability_tab:
        render_reliability()

    with technical_tab:
        render_technical(horizon)


if __name__ == "__main__":
    main()
