"""Tests for how the SHAP helpers treat a residual model disabled by the validation gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxhedge.explain import DISABLED_MESSAGE, compute_shap_values, residual_model_disabled
from fxhedge.model_training import Predictor, TrainedModel, ZeroResidual


class StubTree:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.ones(len(features), dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


def _model(tree: Predictor) -> TrainedModel:
    return TrainedModel(
        kind="hybrid",
        target="y_fx",
        base=None,
        tree=tree,
        feature_columns=("a",),
        params={},
        train_end=pd.Timestamp("2024-01-01"),
    )


def test_zero_residual_is_reported_as_disabled() -> None:
    assert residual_model_disabled(_model(ZeroResidual()))
    assert not residual_model_disabled(_model(StubTree()))


def test_shap_refuses_a_disabled_residual_model() -> None:
    pytest.importorskip("shap")
    features = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        compute_shap_values(_model(ZeroResidual()), features)


def test_disabled_message_explains_the_fallback() -> None:
    assert "Prophet only" in DISABLED_MESSAGE
