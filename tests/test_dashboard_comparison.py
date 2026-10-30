"""Tests for the model comparison table of the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fxhedge.dashboard import comparison_table, load_model_comparison

HORIZONS = (30, 60, 90)


def _record(target: str, model: str, horizon: int, ratio: float | None) -> dict[str, object]:
    return {"target": target, "model": model, "horizon": horizon, "rmse_vs_random_walk": ratio}


def _records() -> list[dict[str, object]]:
    rows = [_record("y_fx", "random_walk", h, 1.0) for h in HORIZONS]
    ratios: dict[str, tuple[float | None, ...]] = {
        "xgboost": (0.892, 0.921, 0.950),
        "hybrid": (1.556, 1.274, 1.146),
        "sarima": (1.007, None, 1.002),
    }
    for model, values in ratios.items():
        rows += [_record("y_fx", model, h, v) for h, v in zip(HORIZONS, values, strict=True)]
    cash = (0.491, 0.423, 0.378)
    rows += [_record("y_cash", "hybrid", h, v) for h, v in zip(HORIZONS, cash, strict=True)]
    return rows


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_model_comparison(tmp_path / "metrics.json") is None


def test_valid_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_records()), encoding="utf-8")
    frame = load_model_comparison(path)
    assert frame is not None
    assert len(frame) == len(_records())


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"target": "y_fx"}', '[{"target": "y_fx"}]'],
)
def test_invalid_files_raise_a_value_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_model_comparison(path)


def test_table_sorts_models_from_best_to_worst_and_formats_values() -> None:
    table = comparison_table(pd.DataFrame(_records()), "y_fx")
    assert list(table.index) == ["XGBoost", "Random walk", "SARIMA", "Hybrid"]
    assert list(table.columns) == ["30 days", "60 days", "90 days"]
    assert table.loc["XGBoost", "30 days"] == "0.89"
    assert table.loc["Hybrid", "90 days"] == "1.15"
    assert table.loc["SARIMA", "60 days"] == "n/a"


def test_table_keeps_each_target_separate() -> None:
    metrics = pd.DataFrame(_records())
    cash = comparison_table(metrics, "y_cash")
    assert list(cash.index) == ["Hybrid"]
    assert cash.loc["Hybrid", "30 days"] == "0.49"
    assert comparison_table(metrics, "unknown").empty
