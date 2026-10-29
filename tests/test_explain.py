"""Tests for SHAP helpers that do not need the shap package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fxhedge.explain import (
    ShapResult,
    describe_drivers,
    mean_abs_shap,
    save_shap_summary,
    top_drivers,
)


def _result() -> ShapResult:
    values = np.array([[1.0, -0.1, 0.0], [-1.0, 0.1, 0.0], [2.0, 0.2, 0.0]])
    return ShapResult(values=values, feature_names=("rate", "month", "flag"), base_value=0.0)


def test_mean_abs_shap_is_sorted_descending() -> None:
    importance = mean_abs_shap(_result())
    assert list(importance.index) == ["rate", "month", "flag"]
    assert importance["rate"] == pytest.approx(4.0 / 3.0)


def test_mean_abs_shap_rejects_inconsistent_shapes() -> None:
    bad = ShapResult(values=np.zeros((3, 2)), feature_names=("a",), base_value=0.0)
    with pytest.raises(ValueError):
        mean_abs_shap(bad)


def test_top_drivers() -> None:
    assert top_drivers(_result(), k=2) == ["rate", "month"]
    with pytest.raises(ValueError):
        top_drivers(_result(), k=0)


def test_describe_drivers_reports_shares() -> None:
    text = describe_drivers(_result(), k=2)
    assert text.startswith("Main drivers")
    assert "rate (91%)" in text


def test_describe_drivers_handles_zero_impact() -> None:
    zero = ShapResult(values=np.zeros((2, 2)), feature_names=("a", "b"), base_value=0.0)
    assert "no measurable impact" in describe_drivers(zero)


def test_save_shap_summary_creates_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "figures" / "shap.png"
    save_shap_summary(_result(), out)
    assert out.exists()
    assert out.stat().st_size > 0
