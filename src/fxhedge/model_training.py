"""Model training: Prophet + LightGBM hybrid, XGBoost comparator and safe serialization.

The hybrid models trend and seasonality with Prophet and lets LightGBM learn the Prophet
residuals from exogenous and calendar features. Residuals are generated out of sample
with a time-series split, so the residual model never learns from in-sample fits.

An optional validation gate keeps the residual model only if it lowers the error on a
holdout of the most recent residuals; otherwise the hybrid falls back to Prophet alone.

Future exogenous values are held at their last known value (carry-forward). Target-derived
lag and rolling columns are excluded from the model features because they are not known
over a 30/60/90-day horizon.

Artifacts are JSON (Prophet, XGBoost) or LightGBM's text format; pickle is never used.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import TimeSeriesSplit

from fxhedge.config import TrainingConfig, load_settings, load_yaml_model
from fxhedge.evaluation import Forecaster
from fxhedge.feature_engineering import add_calendar_features
from fxhedge.io_utils import read_csv_with_date, sha256_file

logger = logging.getLogger(__name__)

Target = Literal["y_fx", "y_cash"]
ModelKind = Literal["hybrid", "xgboost"]
Params = dict[str, float | int]

TARGETS: tuple[Target, ...] = ("y_fx", "y_cash")
BASE_ARTIFACT = "prophet_model.json"
LIGHTGBM_ARTIFACT = "lightgbm_model.txt"
XGBOOST_ARTIFACT = "xgboost_model.json"
METADATA_FILE = "metadata.json"
MIN_GATE_TRAIN_ROWS = 30
GATE_PARAMS: Params = {
    "num_leaves": 8,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
_TARGET_PREFIXES = ("y_fx", "y_cash", "occupancy_rate")
_TRACKED_LIBRARIES = ("prophet", "lightgbm", "xgboost", "optuna", "pandas", "numpy")


class ModelIntegrityError(RuntimeError):
    """Raised when an artifact does not match the hash recorded in metadata.json."""


class BaseForecaster(Protocol):
    """Trend and seasonality model (Prophet)."""

    def predict(self, dates: pd.DatetimeIndex) -> NDArray[np.float64]: ...

    def to_json(self) -> str: ...


class Predictor(Protocol):
    """Tree model that maps a feature table to predictions."""

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]: ...

    def save(self, path: Path) -> None: ...


BaseFitter = Callable[[pd.Series], BaseForecaster]
TreeFitter = Callable[[pd.DataFrame, pd.Series, TrainingConfig], tuple[Predictor, Params]]
BaseLoader = Callable[[Path], BaseForecaster]
TreeLoader = Callable[[Path], Predictor]


@dataclass(frozen=True)
class GateResult:
    """Outcome of the validation gate.

    ``rmse_base`` is the holdout error of the base model alone (the residuals themselves);
    ``rmse_hybrid`` is the holdout error after adding the tree. Both are NaN when there were
    too few rows to validate.
    """

    use_tree: bool
    rmse_base: float
    rmse_hybrid: float
    holdout_rows: int


@dataclass(frozen=True)
class TrainedModel:
    """A fitted model: optional Prophet base plus a tree model over ``feature_columns``."""

    kind: ModelKind
    target: Target
    base: BaseForecaster | None
    tree: Predictor
    feature_columns: tuple[str, ...]
    params: Params
    train_end: pd.Timestamp
    gate: GateResult | None = None


class ZeroResidual:
    """Residual model that adds nothing; used when the validation gate rejects the tree."""

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        return np.zeros(len(features), dtype=np.float64)

    def save(self, path: Path) -> None:
        """Nothing to save: this model has no parameters."""


class ProphetBase:
    """Adapter around a fitted ``prophet.Prophet`` model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, dates: pd.DatetimeIndex) -> NDArray[np.float64]:
        forecast = self._model.predict(pd.DataFrame({"ds": dates}))
        return np.asarray(forecast["yhat"], dtype=np.float64)

    def to_json(self) -> str:
        from prophet.serialize import model_to_json

        return str(model_to_json(self._model))


class TreeModel:
    """Adapter around a fitted LightGBM or XGBoost model (sklearn API or loaded booster)."""

    def __init__(self, estimator: Any, kind: Literal["lightgbm", "xgboost"]) -> None:
        self.estimator = estimator
        self._kind = kind

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        return np.asarray(self.estimator.predict(features), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self._kind == "lightgbm":
            booster = getattr(self.estimator, "booster_", self.estimator)
            booster.save_model(str(path))
        else:
            self.estimator.save_model(str(path))


def fit_prophet_base(series: pd.Series) -> BaseForecaster:
    """Fit Prophet with yearly and weekly seasonality on a date-indexed series."""
    from prophet import Prophet

    frame = pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=np.float64)})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(frame)
    return ProphetBase(model)


def load_prophet_base(path: Path) -> BaseForecaster:
    """Load a Prophet model from its JSON artifact."""
    from prophet.serialize import model_from_json

    return ProphetBase(model_from_json(path.read_text(encoding="utf-8")))


def load_lightgbm(path: Path) -> Predictor:
    """Load a LightGBM booster from its text artifact."""
    import lightgbm as lgb

    return TreeModel(lgb.Booster(model_file=str(path)), "lightgbm")


def load_xgboost(path: Path) -> Predictor:
    """Load an XGBoost model from its JSON artifact."""
    import xgboost as xgb

    estimator = xgb.XGBRegressor()
    estimator.load_model(str(path))
    return TreeModel(estimator, "xgboost")


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns usable over a multi-week horizon: calendar and exogenous features.

    Excludes the date, both targets and every target-derived lag or rolling column.

    Raises:
        ValueError: if no feature column remains.
    """
    excluded = {"date", "y_fx", "y_cash"}
    columns = [
        column
        for column in df.columns
        if column not in excluded and not column.startswith(_TARGET_PREFIXES)
    ]
    if not columns:
        raise ValueError("No feature columns available after excluding targets")
    return columns


def residual_split_count(n_rows: int, cfg: TrainingConfig) -> int:
    """Number of ``residual_test_size`` blocks that fit after the first training window.

    The first training window keeps at least ``residual_min_train`` rows after skipping
    ``cfg.gap`` rows. The result is capped at ``cfg.residual_max_splits`` and is 0 when
    the series is too short for even one block.
    """
    available = (n_rows - cfg.gap - cfg.residual_min_train) // cfg.residual_test_size
    return max(0, min(available, cfg.residual_max_splits))


def oos_residuals(series: pd.Series, fit_base: BaseFitter, cfg: TrainingConfig) -> pd.Series:
    """Out-of-sample residuals of the base model using an expanding time-series split.

    With at least ``cfg.residual_min_splits`` blocks available, each test block has
    ``cfg.residual_test_size`` rows, so the base model is judged over a span close to the
    forecast horizon. Very long test blocks would make Prophet extrapolate its trend for
    more than a year, and the tree would learn a correction for an error that does not
    exist at 30 to 90 days. With fewer rows the split uses ``cfg.n_splits`` as before.

    ``cfg.gap`` rows are skipped between each training window and its test block, which
    mimics forecasting several weeks ahead.
    """
    blocks = residual_split_count(len(series), cfg)
    if blocks >= cfg.residual_min_splits:
        splitter = TimeSeriesSplit(
            n_splits=blocks, gap=cfg.gap, test_size=cfg.residual_test_size
        )
    else:
        splitter = TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.gap)
    parts: list[pd.Series] = []
    for train_index, test_index in splitter.split(series):
        base = fit_base(series.iloc[train_index])
        test_dates = pd.DatetimeIndex(series.index[test_index])
        predicted = base.predict(test_dates)
        actual = series.iloc[test_index].to_numpy(dtype=np.float64)
        parts.append(pd.Series(actual - predicted, index=test_dates))
    combined: pd.Series = pd.concat(parts)
    return combined


def cv_rmse(
    features: pd.DataFrame,
    target: pd.Series,
    make_estimator: Callable[[], Any],
    cfg: TrainingConfig,
) -> float:
    """Mean RMSE of an estimator over a time-series split.

    Raises:
        ValueError: if there are too few rows for the requested splits and gap.
    """
    splitter = TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.gap)
    errors: list[float] = []
    for train_index, test_index in splitter.split(features):
        estimator = make_estimator()
        estimator.fit(features.iloc[train_index], target.iloc[train_index])
        predicted = np.asarray(estimator.predict(features.iloc[test_index]), dtype=np.float64)
        diff = target.iloc[test_index].to_numpy(dtype=np.float64) - predicted
        errors.append(float(np.sqrt(np.mean(diff**2))))
    return float(np.mean(errors))


def build_lightgbm(params: Mapping[str, float | int], seed: int) -> Any:
    """Create an unfitted ``LGBMRegressor`` with the given hyperparameters."""
    import lightgbm as lgb

    estimator_params: dict[str, Any] = {
        **params,
        "subsample_freq": 1,
        "random_state": seed,
        "n_jobs": 1,
        "verbose": -1,
    }
    return lgb.LGBMRegressor(**estimator_params)


def suggest_lightgbm_params(trial: Any) -> Params:
    """Optuna search space for the residual model."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def fit_lightgbm_tuned(
    features: pd.DataFrame, residuals: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, Params]:
    """Tune LightGBM with Optuna (time-series CV RMSE) and refit on all rows."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params = suggest_lightgbm_params(trial)
        return cv_rmse(features, residuals, lambda: build_lightgbm(params, cfg.seed), cfg)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.optuna_trials)
    best: Params = dict(study.best_params)
    estimator = build_lightgbm(best, cfg.seed)
    estimator.fit(features, residuals)
    return TreeModel(estimator, "lightgbm"), best


def fit_lightgbm_fixed(
    features: pd.DataFrame, residuals: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, Params]:
    """Fit LightGBM with fixed, conservative settings (no tuning). Used by the gate."""
    estimator = build_lightgbm(GATE_PARAMS, cfg.seed)
    estimator.fit(features, residuals)
    return TreeModel(estimator, "lightgbm"), dict(GATE_PARAMS)


def fit_xgboost_direct(
    features: pd.DataFrame, target: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, Params]:
    """Fit an XGBoost regressor directly on the target level."""
    import xgboost as xgb

    params: Params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    estimator = xgb.XGBRegressor(**params, random_state=cfg.seed, n_jobs=1)
    estimator.fit(features, target)
    return TreeModel(estimator, "xgboost"), params


def _prepare(df: pd.DataFrame, target: Target) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    data = df.sort_values("date").set_index("date")
    series = data[target].astype(float)
    return data, series, select_feature_columns(data.reset_index())


def evaluate_gate(
    features: pd.DataFrame, residuals: pd.Series, cfg: TrainingConfig, fitter: TreeFitter
) -> GateResult:
    """Decide whether a residual tree helps, using the most recent residuals as holdout.

    The tree is fitted on the earlier residuals, ``cfg.gap`` rows are skipped, and its
    predictions are compared with the holdout residuals. Using only the base model means
    predicting a residual of zero, so the tree is kept only when its holdout RMSE is lower
    than that of the residuals themselves by more than ``cfg.gate_min_improvement``.

    With fewer than ``MIN_GATE_TRAIN_ROWS`` rows left for fitting, the tree cannot be
    validated and is rejected.
    """
    total = len(residuals)
    holdout = max(1, round(total * cfg.gate_holdout_fraction))
    train_rows = total - holdout - cfg.gap
    if train_rows < MIN_GATE_TRAIN_ROWS:
        return GateResult(False, float("nan"), float("nan"), holdout)
    tree, _ = fitter(features.iloc[:train_rows], residuals.iloc[:train_rows], cfg)
    actual = residuals.iloc[total - holdout :].to_numpy(dtype=np.float64)
    predicted = tree.predict(features.iloc[total - holdout :])
    rmse_base = float(np.sqrt(np.mean(actual**2)))
    rmse_hybrid = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    use_tree = rmse_hybrid < rmse_base * (1.0 - cfg.gate_min_improvement)
    return GateResult(use_tree, rmse_base, rmse_hybrid, holdout)


def fit_hybrid(
    df: pd.DataFrame,
    target: Target,
    cfg: TrainingConfig,
    base_fitter: BaseFitter = fit_prophet_base,
    residual_fitter: TreeFitter = fit_lightgbm_tuned,
    gate_fitter: TreeFitter | None = None,
) -> TrainedModel:
    """Fit the Prophet + tree hybrid for ``target``.

    When ``gate_fitter`` is given, a validation gate runs first (see ``evaluate_gate``).
    If it rejects the tree, the model keeps Prophet alone and ``residual_fitter`` is not
    called. Without ``gate_fitter`` the tree is always fitted.
    """
    data, series, columns = _prepare(df, target)
    residuals = oos_residuals(series, base_fitter, cfg)
    features = data.loc[residuals.index, columns]
    gate = evaluate_gate(features, residuals, cfg, gate_fitter) if gate_fitter else None
    tree: Predictor
    params: Params
    if gate is not None and not gate.use_tree:
        logger.warning(
            "Validation gate rejected the residual tree for %s (holdout RMSE %.4f with the "
            "tree vs %.4f without it); using Prophet alone",
            target,
            gate.rmse_hybrid,
            gate.rmse_base,
        )
        tree, params = ZeroResidual(), {}
    else:
        tree, params = residual_fitter(features, residuals, cfg)
    return TrainedModel(
        kind="hybrid",
        target=target,
        base=base_fitter(series),
        tree=tree,
        feature_columns=tuple(columns),
        params=params,
        train_end=pd.Timestamp(series.index[-1]),
        gate=gate,
    )


def fit_xgboost(
    df: pd.DataFrame,
    target: Target,
    cfg: TrainingConfig,
    tree_fitter: TreeFitter = fit_xgboost_direct,
) -> TrainedModel:
    """Fit the XGBoost comparator that predicts the target level directly."""
    data, series, columns = _prepare(df, target)
    tree, params = tree_fitter(data[columns], series, cfg)
    return TrainedModel(
        kind="xgboost",
        target=target,
        base=None,
        tree=tree,
        feature_columns=tuple(columns),
        params=params,
        train_end=pd.Timestamp(series.index[-1]),
    )


def future_feature_frame(
    history: pd.DataFrame, dates: pd.DatetimeIndex, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Features for future dates: calendar recomputed, other columns carried forward."""
    last = history.sort_values("date").iloc[-1]
    frame = add_calendar_features(pd.DataFrame({"date": dates}))
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = float(last[column])
    return frame.set_index("date")[list(feature_columns)]


def predict_values(model: TrainedModel, features: pd.DataFrame) -> NDArray[np.float64]:
    """Predict the target for a date-indexed feature table."""
    table = features.loc[:, list(model.feature_columns)]
    tree_part = model.tree.predict(table)
    if model.base is None:
        return tree_part
    return model.base.predict(pd.DatetimeIndex(table.index)) + tree_part


def make_model_forecaster(
    df: pd.DataFrame,
    target: Target,
    cfg: TrainingConfig,
    kind: ModelKind = "hybrid",
    base_fitter: BaseFitter = fit_prophet_base,
    residual_fitter: TreeFitter = fit_lightgbm_tuned,
    xgboost_fitter: TreeFitter = fit_xgboost_direct,
    gate_fitter: TreeFitter | None = None,
) -> Forecaster:
    """Wrap a model as a backtest forecaster (refits on each fold; can be slow)."""

    def forecaster(train: pd.Series, horizon: int) -> pd.Series:
        last = pd.Timestamp(train.index[-1])
        history = df[df["date"] <= last]
        model = (
            fit_hybrid(history, target, cfg, base_fitter, residual_fitter, gate_fitter)
            if kind == "hybrid"
            else fit_xgboost(history, target, cfg, xgboost_fitter)
        )
        dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
        features = future_feature_frame(history, dates, model.feature_columns)
        return pd.Series(predict_values(model, features), index=dates)

    return forecaster


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _clean_float(value: float) -> float | None:
    return None if math.isnan(value) else value


def _gate_info(gate: GateResult | None) -> dict[str, object] | None:
    if gate is None:
        return None
    return {
        "use_tree": gate.use_tree,
        "rmse_base": _clean_float(gate.rmse_base),
        "rmse_hybrid": _clean_float(gate.rmse_hybrid),
        "holdout_rows": gate.holdout_rows,
    }


def save_model(model: TrainedModel, out_dir: Path) -> None:
    """Write model artifacts and ``metadata.json`` (with SHA-256 hashes) to ``out_dir``.

    A model whose residual tree was rejected by the gate stores no tree artifact and
    records ``"residual_model": "none"`` in the metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if model.base is not None:
        (out_dir / BASE_ARTIFACT).write_text(model.base.to_json(), encoding="utf-8")
        artifacts[BASE_ARTIFACT] = sha256_file(out_dir / BASE_ARTIFACT)
    tree_name = LIGHTGBM_ARTIFACT if model.kind == "hybrid" else XGBOOST_ARTIFACT
    disabled = isinstance(model.tree, ZeroResidual)
    if disabled:
        (out_dir / tree_name).unlink(missing_ok=True)
    else:
        model.tree.save(out_dir / tree_name)
        artifacts[tree_name] = sha256_file(out_dir / tree_name)
    residual_model = "none" if disabled else ("lightgbm" if model.kind == "hybrid" else "xgboost")
    info = {
        "kind": model.kind,
        "target": model.target,
        "feature_columns": list(model.feature_columns),
        "params": model.params,
        "train_end": model.train_end.date().isoformat(),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "library_versions": _library_versions(),
        "residual_model": residual_model,
        "gate": _gate_info(model.gate),
        "artifacts": artifacts,
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_trained_model(
    model_dir: Path,
    base_loader: BaseLoader = load_prophet_base,
    tree_loader: TreeLoader | None = None,
) -> TrainedModel:
    """Load a model after verifying every artifact against its recorded SHA-256.

    Raises:
        ModelIntegrityError: if an artifact is missing or has been modified.
    """
    info = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    for name, expected in info["artifacts"].items():
        path = model_dir / name
        if not path.exists() or sha256_file(path) != expected:
            raise ModelIntegrityError(f"Artifact failed integrity check: {name}")
    kind: ModelKind = info["kind"]
    if tree_loader is None:
        tree_loader = load_lightgbm if kind == "hybrid" else load_xgboost
    tree_name = LIGHTGBM_ARTIFACT if kind == "hybrid" else XGBOOST_ARTIFACT
    base = base_loader(model_dir / BASE_ARTIFACT) if kind == "hybrid" else None
    tree: Predictor = (
        ZeroResidual()
        if info.get("residual_model") == "none"
        else tree_loader(model_dir / tree_name)
    )
    return TrainedModel(
        kind=kind,
        target=info["target"],
        base=base,
        tree=tree,
        feature_columns=tuple(info["feature_columns"]),
        params=info["params"],
        train_end=pd.Timestamp(info["train_end"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: train and save the hybrid and the XGBoost comparator."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Train forecasting models.")
    parser.add_argument(
        "--master", type=Path, default=settings.data_dir / "processed" / "master_features.csv"
    )
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Always keep the residual tree, without the validation gate.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_yaml_model(settings.configs_dir / "training.yaml", TrainingConfig)
    master = read_csv_with_date(args.master)
    gate_fitter = None if args.no_gate else fit_lightgbm_fixed
    for target in TARGETS:
        logger.info("Training hybrid model for %s", target)
        hybrid = fit_hybrid(master, target, cfg, gate_fitter=gate_fitter)
        save_model(hybrid, args.models_dir / target)
        if not args.skip_xgboost:
            logger.info("Training XGBoost comparator for %s", target)
            save_model(fit_xgboost(master, target, cfg), args.models_dir / f"{target}_xgboost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
