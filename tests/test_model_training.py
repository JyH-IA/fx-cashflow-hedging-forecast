"""Tests for model training with lightweight fakes for Prophet and the tree models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from fxhedge.config import TrainingConfig
from fxhedge.feature_engineering import add_calendar_features
from fxhedge.model_training import (
    BASE_ARTIFACT,
    METADATA_FILE,
    BaseForecaster,
    ModelIntegrityError,
    Predictor,
    cv_rmse,
    fit_hybrid,
    fit_xgboost,
    future_feature_frame,
    load_trained_model,
    make_model_forecaster,
    oos_residuals,
    predict_values,
    save_model,
    select_feature_columns,
)

CFG = TrainingConfig(n_splits=3, gap=5, optuna_trials=1)


class LinearBase:
    def __init__(self, intercept: float, slope: float, origin: pd.Timestamp) -> None:
        self.intercept = intercept
        self.slope = slope
        self.origin = origin

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        days = np.asarray((dates - self.origin).days, dtype=np.float64)
        return self.intercept + self.slope * days

    def to_json(self) -> str:
        return json.dumps(
            {"intercept": self.intercept, "slope": self.slope, "origin": str(self.origin)}
        )


def fit_linear_base(series: pd.Series) -> BaseForecaster:
    days = np.asarray((series.index - series.index[0]).days, dtype=np.float64)
    slope, intercept = np.polyfit(days, series.to_numpy(dtype=np.float64), 1)
    return LinearBase(float(intercept), float(slope), pd.Timestamp(series.index[0]))


def load_linear_base(path: Path) -> BaseForecaster:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LinearBase(data["intercept"], data["slope"], pd.Timestamp(data["origin"]))


class RidgeTree:
    def __init__(self, coef: np.ndarray, intercept: float) -> None:
        self.coef = coef
        self.intercept = intercept
        self.estimator = self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features.to_numpy(dtype=np.float64) @ self.coef + self.intercept

    def save(self, path: Path) -> None:
        payload = {"coef": self.coef.tolist(), "intercept": self.intercept}
        path.write_text(json.dumps(payload), encoding="utf-8")


def fit_ridge(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, dict[str, float | int]]:
    model = Ridge(alpha=1.0).fit(features, target)
    return RidgeTree(np.asarray(model.coef_), float(model.intercept_)), {"alpha": 1.0}


def load_ridge(path: Path) -> Predictor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RidgeTree(np.asarray(data["coef"]), float(data["intercept"]))


def _frame(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    rng = np.random.default_rng(0)
    elapsed = np.arange(days, dtype=np.float64)
    frame["cetes_28d"] = 10.0 + rng.normal(0.0, 0.01, days)
    frame["y_fx"] = 17.0 + 0.002 * elapsed + 0.4 * frame["is_weekend"] + rng.normal(0, 0.01, days)
    frame["y_cash"] = 100.0 + 0.1 * elapsed + 5.0 * frame["is_weekend"]
    frame["y_fx_lag7"] = frame["y_fx"].shift(7).bfill()
    return frame


def test_select_feature_columns_excludes_targets_and_lags() -> None:
    columns = select_feature_columns(_frame(30))
    assert "cetes_28d" in columns
    assert "is_weekend" in columns
    for excluded in ("date", "y_fx", "y_cash", "y_fx_lag7"):
        assert excluded not in columns


def test_select_feature_columns_requires_features() -> None:
    with pytest.raises(ValueError):
        select_feature_columns(pd.DataFrame({"date": [1], "y_fx": [1.0]}))


def test_oos_residuals_cover_only_test_blocks() -> None:
    frame = _frame(400)
    series = frame.set_index("date")["y_fx"]
    residuals = oos_residuals(series, fit_linear_base, CFG)
    assert 0 < len(residuals) < len(series)
    assert residuals.index.min() > series.index[0]
    assert residuals.index.is_unique


def test_cv_rmse_returns_finite_value_and_rejects_short_data() -> None:
    frame = _frame(300)
    features = frame[["is_weekend", "cetes_28d"]]
    value = cv_rmse(features, frame["y_fx"], lambda: Ridge(alpha=1.0), CFG)
    assert np.isfinite(value)
    with pytest.raises(ValueError):
        cv_rmse(features.head(6), frame["y_fx"].head(6), lambda: Ridge(), CFG)


def test_hybrid_improves_on_base_with_calendar_effect() -> None:
    frame = _frame(800)
    history, holdout = frame.iloc[:740], frame.iloc[740:]
    model = fit_hybrid(history, "y_fx", CFG, fit_linear_base, fit_ridge)
    dates = pd.DatetimeIndex(holdout["date"])
    features = future_feature_frame(history, dates, model.feature_columns)
    hybrid = predict_values(model, features)
    base_only = model.base.predict(dates) if model.base is not None else hybrid
    truth = holdout["y_fx"].to_numpy(dtype=np.float64)
    hybrid_rmse = float(np.sqrt(np.mean((truth - hybrid) ** 2)))
    base_rmse = float(np.sqrt(np.mean((truth - base_only) ** 2)))
    assert hybrid_rmse < 0.5 * base_rmse


def test_xgboost_comparator_has_no_base() -> None:
    model = fit_xgboost(_frame(300), "y_cash", CFG, fit_ridge)
    assert model.kind == "xgboost"
    assert model.base is None
    features = future_feature_frame(
        _frame(300), pd.date_range("2023-01-01", periods=5, freq="D"), model.feature_columns
    )
    assert predict_values(model, features).shape == (5,)


def test_future_feature_frame_recomputes_calendar_and_carries_exogenous() -> None:
    history = _frame(100)
    dates = pd.date_range("2022-04-12", periods=4, freq="D")
    frame = future_feature_frame(history, dates, ["cetes_28d", "is_weekend", "month"])
    assert (frame["cetes_28d"] == float(history["cetes_28d"].iloc[-1])).all()
    assert frame["month"].tolist() == [4, 4, 4, 4]
    assert frame.index.equals(dates)


def test_save_and_load_roundtrip_verifies_integrity(tmp_path: Path) -> None:
    frame = _frame(500)
    model = fit_hybrid(frame, "y_fx", CFG, fit_linear_base, fit_ridge)
    save_model(model, tmp_path / "y_fx")
    info = json.loads((tmp_path / "y_fx" / METADATA_FILE).read_text(encoding="utf-8"))
    assert set(info["artifacts"]) == {BASE_ARTIFACT, "lightgbm_model.txt"}
    assert info["target"] == "y_fx"

    loaded = load_trained_model(tmp_path / "y_fx", load_linear_base, load_ridge)
    dates = pd.date_range("2023-06-01", periods=10, freq="D")
    features = future_feature_frame(frame, dates, model.feature_columns)
    assert np.allclose(predict_values(model, features), predict_values(loaded, features))

    (tmp_path / "y_fx" / BASE_ARTIFACT).write_text("{}", encoding="utf-8")
    with pytest.raises(ModelIntegrityError):
        load_trained_model(tmp_path / "y_fx", load_linear_base, load_ridge)


def test_model_forecaster_returns_future_path() -> None:
    frame = _frame(500)
    forecaster = make_model_forecaster(frame, "y_fx", CFG, "hybrid", fit_linear_base, fit_ridge)
    train = frame.set_index("date")["y_fx"].iloc[:300]
    path = forecaster(train, 7)
    assert len(path) == 7
    assert path.index[0] == train.index[-1] + pd.Timedelta(days=1)
    assert np.isfinite(path.to_numpy()).all()


def test_model_forecaster_supports_xgboost_kind() -> None:
    frame = _frame(500)
    forecaster = make_model_forecaster(
        frame, "y_cash", CFG, "xgboost", xgboost_fitter=fit_ridge
    )
    train = frame.set_index("date")["y_cash"].iloc[:300]
    assert len(forecaster(train, 5)) == 5


def test_lightgbm_tuning_runs_when_dependencies_are_installed() -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    from fxhedge.model_training import fit_lightgbm_tuned

    frame = _frame(400)
    features = frame[["is_weekend", "cetes_28d"]]
    residuals = frame["y_fx"] - frame["y_fx"].mean()
    tree, params = fit_lightgbm_tuned(features, residuals, CFG)
    assert tree.predict(features).shape == (len(features),)
    assert "num_leaves" in params
