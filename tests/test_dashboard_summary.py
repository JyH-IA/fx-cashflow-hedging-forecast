"""Tests for the headline, key figures and plain-language helpers of the dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.dashboard import (
    DEFAULT_RATIOS,
    HEDGE_LEVELS,
    base_only_targets,
    format_compact_money,
    friendly_impact_table,
    hedge_ratio,
    reliability_verdicts,
    summary_headline,
    summary_kpis,
)
from fxhedge.model_training import Predictor, TrainedModel, ZeroResidual

HORIZONS = (30, 60, 90)


class StubBase:
    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.zeros(len(dates), dtype=np.float64)

    def to_json(self) -> str:
        return "{}"


class StubTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.ones(len(features), dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


def _impact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hedge_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            "forward_rate": [10.2954] * 5,
            "expected_proceeds_mxn": [
                166_575_250.0,
                167_077_913.0,
                167_580_830.0,
                168_083_747.0,
                168_586_665.0,
            ],
            "worst_case_mxn": [
                148_531_233.0,
                153_544_928.0,
                158_558_849.0,
                163_572_771.0,
                168_586_634.0,
            ],
            "best_case_mxn": [
                186_474_089.0,
                182_002_012.0,
                177_530_219.0,
                173_058_427.0,
                168_586_692.0,
            ],
            "range_mxn": [37_942_856.0, 28_457_084.0, 18_971_370.0, 9_485_656.0, 58.0],
            "bank_cost_mxn": [0.0, 63_569.0, 126_884.0, 190_199.0, 253_514.0],
        }
    )


def _record(target: str, model: str, horizon: int, ratio: float) -> dict[str, object]:
    return {"target": target, "model": model, "horizon": horizon, "rmse_vs_random_walk": ratio}


def _metrics() -> pd.DataFrame:
    ratios: dict[tuple[str, str], tuple[float, ...]] = {
        ("y_fx", "random_walk"): (1.0, 1.0, 1.0),
        ("y_fx", "seasonal_naive_7"): (1.020, 1.006, 1.033),
        ("y_fx", "xgboost"): (0.892, 0.921, 0.950),
        ("y_fx", "sarima"): (1.007, 1.005, 1.002),
        ("y_fx", "hybrid"): (1.556, 1.274, 1.146),
        ("y_cash", "random_walk"): (1.0, 1.0, 1.0),
        ("y_cash", "seasonal_naive_365"): (0.629, 0.569, 0.531),
        ("y_cash", "hybrid"): (0.491, 0.423, 0.378),
        ("y_cash", "sarima"): (0.759, 0.804, 0.877),
        ("y_cash", "xgboost"): (0.795, 0.826, 0.866),
    }
    rows = [
        _record(target, model, horizon, value)
        for (target, model), values in ratios.items()
        for horizon, value in zip(HORIZONS, values, strict=True)
    ]
    return pd.DataFrame(rows)


def test_hedge_levels_match_the_default_ratios() -> None:
    assert tuple(ratio for _, ratio in HEDGE_LEVELS) == DEFAULT_RATIOS


def test_hedge_ratio_reads_a_label_and_rejects_an_unknown_one() -> None:
    assert hedge_ratio("Balanced (50%)") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        hedge_ratio("Extreme")


@pytest.mark.parametrize(
    ("value", "currency", "expected"),
    [
        (9_600_000.0, "MXN", "$9.6M MXN"),
        (127_000.0, "MXN", "$127.0K MXN"),
        (950.0, "USD", "$950 USD"),
        (-2_500_000.0, "MXN", "-$2.5M MXN"),
        (1_200_000_000.0, "USD", "$1.2B USD"),
        (0.0, "MXN", "$0 MXN"),
        (999_999.0, "MXN", "$1.0M MXN"),
    ],
)
def test_compact_money(value: float, currency: str, expected: str) -> None:
    assert format_compact_money(value, currency) == expected


def test_kpis_compare_the_selected_hedge_with_no_hedge() -> None:
    kpis = summary_kpis(_impact(), 0.5)
    assert kpis["protected"] == pytest.approx(158_558_849.0 - 148_531_233.0)
    assert kpis["narrowing_pct"] == pytest.approx(50.0, abs=0.01)
    assert kpis["bank_cost"] == pytest.approx(126_884.0)
    assert summary_kpis(_impact(), 0.0)["protected"] == pytest.approx(0.0)


def test_kpis_need_the_no_hedge_row_and_the_selected_ratio() -> None:
    with pytest.raises(ValueError):
        summary_kpis(_impact().iloc[1:], 0.5)
    with pytest.raises(ValueError):
        summary_kpis(_impact(), 0.4)


def test_headline_states_what_the_hedge_protects() -> None:
    kpis = summary_kpis(_impact(), 0.5)
    text = summary_headline(kpis, 90, 0.5, 16_400_000.0)
    assert "Hedging 50%" in text
    assert "3 months" in text
    assert "$16.4M USD" in text
    assert "$10.0M MXN" in text
    assert "50%" in text
    assert "$126.9K MXN" in text


def test_headline_without_a_hedge_explains_the_bad_case() -> None:
    kpis = summary_kpis(_impact(), 0.0)
    text = summary_headline(kpis, 30, 0.0, 16_400_000.0)
    assert "Without a hedge" in text
    assert "next month" in text
    assert "$148.5M MXN" in text


def test_friendly_table_uses_plain_labels_and_compact_amounts() -> None:
    table = friendly_impact_table(_impact())
    assert list(table.columns) == [
        "Hedge level",
        "Forward rate",
        "Expected income",
        "Bad case",
        "Good case",
        "Bank cost",
    ]
    row = table.loc[table["Hedge level"] == "Balanced (50%)"].iloc[0]
    assert row["Bad case"] == "$158.6M MXN"
    assert row["Bank cost"] == "$126.9K MXN"
    assert row["Forward rate"] == "10.2954"


def test_verdicts_separate_a_clear_gain_from_a_modest_one() -> None:
    verdicts = reliability_verdicts(_metrics())
    assert "XGBoost is modestly better" in verdicts["y_fx"]
    assert "(Random walk)" in verdicts["y_fx"]
    assert "about 8% less error" in verdicts["y_fx"]
    assert "Hybrid is clearly better" in verdicts["y_cash"]
    assert "(Seasonal naive (365 days))" in verdicts["y_cash"]
    assert "about 25% less error" in verdicts["y_cash"]


def test_verdict_says_so_when_no_trained_model_wins() -> None:
    metrics = pd.DataFrame(
        [_record("y_fx", "random_walk", 30, 1.0), _record("y_fx", "xgboost", 30, 0.99)]
    )
    assert "no trained model clearly beats" in reliability_verdicts(metrics)["y_fx"]


def test_verdicts_skip_targets_without_trained_models() -> None:
    metrics = _metrics()
    only_baselines = metrics[metrics["model"].isin(["random_walk", "seasonal_naive_365"])]
    assert reliability_verdicts(only_baselines) == {}


def _model(target: str, tree: Predictor) -> TrainedModel:
    return TrainedModel(
        kind="hybrid",
        target="y_fx" if target == "y_fx" else "y_cash",
        base=StubBase(),
        tree=tree,
        feature_columns=("a",),
        params={},
        train_end=pd.Timestamp("2026-09-30"),
    )


def test_base_only_targets_lists_the_targets_without_a_tree() -> None:
    models = {
        "y_fx": _model("y_fx", ZeroResidual()),
        "y_cash": _model("y_cash", StubTree()),
    }
    assert base_only_targets(models) == ["y_fx"]
