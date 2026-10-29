"""Tests for monitoring."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.config import MonitoringConfig
from fxhedge.monitoring import (
    MonitoringReport,
    build_monitoring_report,
    population_stability_index,
    retrain_if_needed,
    retrain_reasons,
    rolling_error,
    run_monitoring,
    should_retrain,
    write_monitoring_report,
)

CFG = MonitoringConfig(rolling_window=30)


def _master(days: int = 600, shift_last: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rate = rng.normal(10.0, 1.0, days)
    rate[-90:] += shift_last
    return pd.DataFrame(
        {"date": dates, "cetes_28d": rate, "month": dates.month, "flag": np.zeros(days)}
    )


def test_psi_is_near_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0.0, 1.0, 5000))
    current = pd.Series(rng.normal(0.0, 1.0, 5000))
    assert population_stability_index(reference, current) < 0.05


def test_psi_is_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0.0, 1.0, 5000))
    shifted = pd.Series(rng.normal(2.0, 1.0, 5000))
    assert population_stability_index(reference, shifted) > 0.5


def test_psi_handles_discrete_and_constant_columns() -> None:
    same = pd.Series([1.0] * 50)
    assert population_stability_index(same, same) == pytest.approx(0.0, abs=1e-9)
    assert population_stability_index(same, pd.Series([2.0] * 50)) > 1.0


@pytest.mark.parametrize(
    ("reference", "current", "bins"),
    [([], [1.0], 10), ([1.0], [], 10), ([1.0, 2.0], [1.0], 1)],
    ids=["empty-reference", "empty-current", "few-bins"],
)
def test_psi_validates_inputs(reference: list[float], current: list[float], bins: int) -> None:
    with pytest.raises(ValueError):
        population_stability_index(
            pd.Series(reference, dtype=float), pd.Series(current, dtype=float), bins
        )


def test_rolling_error_values_and_validation() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    actual = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=index)
    predicted = pd.Series([11.0, 9.0, 11.0, 9.0, 12.0], index=index)
    result = rolling_error(actual, predicted, window=2)
    assert list(result.columns) == ["date", "error", "rolling_rmse", "rolling_mape"]
    assert np.isnan(result["rolling_rmse"].iloc[0])
    assert result["rolling_rmse"].iloc[1] == pytest.approx(1.0)
    assert result["rolling_mape"].iloc[1] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        rolling_error(actual, predicted.head(3), window=2)
    with pytest.raises(ValueError):
        rolling_error(actual, predicted, window=1)


def test_report_flags_drifted_variable_only() -> None:
    report = run_monitoring(_master(shift_last=3.0), None, CFG)
    assert report.drifted_features == ["cetes_28d"]
    assert report.status["cetes_28d"] == "red"
    assert report.status["flag"] == "green"
    assert "month" not in report.psi
    assert report.error_ratio is None
    assert should_retrain(report, CFG)


def test_stable_data_does_not_trigger_retraining() -> None:
    report = run_monitoring(_master(), None, CFG)
    assert report.drifted_features == []
    assert not should_retrain(report, CFG)
    assert retrain_reasons(report, CFG) == []


def test_error_degradation_triggers_retraining() -> None:
    master = _master()
    dates = master["date"]
    rng = np.random.default_rng(2)
    error = rng.normal(0.0, 1.0, len(dates))
    error[-30:] *= 4.0
    errors = pd.DataFrame({"date": dates, "error": error})
    report = run_monitoring(master, errors, CFG)
    assert report.error_ratio is not None
    assert report.error_ratio > CFG.error_ratio_threshold
    assert report.drifted_features == []
    assert any("recent error" in reason for reason in retrain_reasons(report, CFG))


def test_run_monitoring_requires_both_windows() -> None:
    with pytest.raises(ValueError):
        run_monitoring(_master(days=60), None, CFG, current_days=90)


def test_build_report_ignores_non_shared_and_non_numeric_columns() -> None:
    reference = _master().assign(label="a")
    current = _master().drop(columns=["flag"])
    report = build_monitoring_report(reference, current, None, CFG)
    assert "flag" not in report.psi
    assert "label" not in report.psi


def test_retrain_if_needed_calls_retrain_only_when_required() -> None:
    calls: list[int] = []

    def fake_retrain() -> int:
        calls.append(1)
        return 0

    healthy = MonitoringReport(psi={"a": 0.01}, status={"a": "green"}, drifted_features=[])
    drifted = MonitoringReport(psi={"a": 0.5}, status={"a": "red"}, drifted_features=["a"])
    assert retrain_if_needed(healthy, CFG, fake_retrain) is False
    assert calls == []
    assert retrain_if_needed(drifted, CFG, fake_retrain) is True
    assert calls == [1]


def test_write_monitoring_report_creates_dated_json(tmp_path: Path) -> None:
    report = MonitoringReport(psi={"a": 0.01}, status={"a": "green"}, drifted_features=[])
    path = write_monitoring_report(report, tmp_path / "monitoring", date(2026, 10, 29))
    assert path.name == "2026-10-29.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["psi"] == {"a": 0.01}
    assert data["error_ratio"] is None
