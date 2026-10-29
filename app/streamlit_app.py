"""Executive dashboard for the FX hedging forecast (Streamlit). Run with ``make app``.

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
    commission_saving_mxn,
    forecast_summary,
    format_mxn,
    impact_table,
)
from fxhedge.inference import Scenario, load_models, predict_forecast
from fxhedge.io_utils import read_csv_with_date
from fxhedge.model_training import ModelIntegrityError

SCENARIOS = ("base", "alza", "baja")


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
            name="Prediction interval",
        )
    )
    figure.add_trace(go.Scatter(x=frame["date"], y=frame["fx_predicted"], name="Forecast"))
    figure.update_layout(
        title="USD/MXN forecast",
        yaxis_title="MXN per USD",
        xaxis_title="Date",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    """Human-readable copy of the impact table."""
    shown = pd.DataFrame({"Hedge ratio": (table["hedge_ratio"] * 100).map("{:.0f}%".format)})
    shown["Forward rate"] = table["forward_rate"].map("{:.4f}".format)
    labels = {
        "expected_proceeds_mxn": "Expected proceeds",
        "worst_case_mxn": "Worst case",
        "best_case_mxn": "Best case",
        "range_mxn": "Range",
        "expected_benefit_mxn": "Expected benefit",
        "bank_cost_mxn": "Bank cost",
    }
    for column, label in labels.items():
        shown[label] = table[column].map(format_mxn)
    return shown


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="FX hedging forecast", layout="wide")
    st.title("USD/MXN and cash-flow forecast")
    st.warning(DISCLAIMER)

    horizon = int(st.sidebar.selectbox("Horizon (days)", HORIZONS, index=0))
    scenario = str(st.sidebar.selectbox("Scenario", SCENARIOS, index=0))
    run_date = st.sidebar.date_input("Run date", value=date.today())

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
    left, middle, right = st.columns(3)
    left.metric("Last spot", f"{summary['spot']:.4f}")
    middle.metric(
        "Forecast at horizon",
        f"{summary['settlement_fx']:.4f}",
        f"{summary['change_pct']:+.2f}%",
    )
    right.metric("Expected USD inflow", f"${summary['expected_cash_usd']:,.0f}")
    st.plotly_chart(forecast_chart(frame))

    st.subheader("Hedge simulation")
    expected = summary["expected_cash_usd"]
    notional = float(
        st.sidebar.slider(
            "Amount to protect (USD)",
            min_value=0.0,
            max_value=max(expected * 2.0, 1.0),
            value=float(expected),
            step=max(round(expected / 100.0), 1.0),
        )
    )
    ratio = st.sidebar.slider("Selected hedge ratio (%)", 0, 100, 50) / 100.0
    table = impact_table(frame, spot, notional, DEFAULT_RATIOS, business, horizon)
    st.dataframe(display_table(table), hide_index=True)

    chosen = impact_table(frame, spot, notional, [ratio], business, horizon).iloc[0]
    saving = commission_saving_mxn(notional, ratio, spot, business, horizon)
    first, second, third = st.columns(3)
    first.metric("Worst case (selected)", format_mxn(float(chosen["worst_case_mxn"])))
    second.metric("Bank cost (selected)", format_mxn(float(chosen["bank_cost_mxn"])))
    third.metric("Commission avoided vs 100% hedge", format_mxn(saving))
    st.caption(
        "Assumptions: forward priced with covered interest parity; "
        "bank spread and fixed fee from configs/business.yaml; "
        "exogenous variables held at their last known value."
    )


if __name__ == "__main__":
    main()
