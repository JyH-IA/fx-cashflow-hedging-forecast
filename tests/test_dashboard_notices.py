"""Tests for the notice about targets whose forecast comes from the base model alone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fxhedge.dashboard import base_only_notices
from fxhedge.model_training import (
    BaseForecaster,
    LastValueBase,
    Predictor,
    Target,
    TrainedModel,
    ZeroResidual,
)


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


def _model(target: Target, base: BaseForecaster, tree: Predictor) -> TrainedModel:
    return TrainedModel(
        kind="hybrid",
        target=target,
        base=base,
        tree=tree,
        feature_columns=("a",),
        params={},
        train_end=pd.Timestamp("2026-09-30"),
    )


def test_random_walk_base_without_a_tree_is_reported_as_flat() -> None:
    base = LastValueBase(17.5, pd.Timestamp("2026-09-30"))
    notices = base_only_notices({"y_fx": _model("y_fx", base, ZeroResidual())})
    assert len(notices) == 1
    assert "USD/MXN" in notices[0]
    assert "flat" in notices[0]
    assert "scenario selector" in notices[0]


def test_prophet_style_base_without_a_tree_is_not_called_flat() -> None:
    notices = base_only_notices({"y_cash": _model("y_cash", StubBase(), ZeroResidual())})
    assert len(notices) == 1
    assert "USD cash-in" in notices[0]
    assert "trend and seasonality" in notices[0]
    assert "flat" not in notices[0]


def test_targets_that_keep_their_tree_get_no_notice() -> None:
    models = {
        "y_fx": _model("y_fx", StubBase(), StubTree()),
        "y_cash": _model("y_cash", StubBase(), ZeroResidual()),
    }
    notices = base_only_notices(models)
    assert len(notices) == 1
    assert notices[0].startswith("USD cash-in")
