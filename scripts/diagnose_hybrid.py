"""Diagnose the Prophet + LightGBM hybrid on the backtest origins.

For each origin the script fits the hybrid on the history up to that date and compares the
forecast of the next ``horizon`` days with the real values. It prints where the error comes
from, so a bad result can be traced to a cause instead of guessed.

How to read the columns:

- ``bias_base``: mean of (actual - Prophet). A large value means the Prophet trend is
  already off at the origin.
- ``tree_mean``: mean correction added by LightGBM. If ``bias_hybrid`` is larger than
  ``bias_base``, the tree made the forecast worse.
- ``resid_mean_train`` and ``resid_std_train``: out-of-sample residuals used to train the
  tree. A mean far from zero means the tree learned a level that the final Prophet fit does
  not share.
- ``features_out_of_range``: features whose carried-forward value lies outside the range seen
  while training the tree. Trees cannot extrapolate beyond that range.

Run it from the repository root after ``make demo``::

    python scripts/diagnose_hybrid.py --target y_fx --horizon 30

It uses an untuned LightGBM so it runs much faster than the full backtest.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fxhedge.config import TrainingConfig, load_settings, load_yaml_model
from fxhedge.evaluation import MODEL_MIN_TRAIN_ROWS, fold_origins
from fxhedge.io_utils import read_csv_with_date
from fxhedge.model_training import (
    Params,
    Predictor,
    TreeModel,
    build_lightgbm,
    fit_hybrid,
    fit_prophet_base,
    future_feature_frame,
    oos_residuals,
)

logger = logging.getLogger(__name__)


def fit_quick_lightgbm(
    features: pd.DataFrame, residuals: pd.Series, cfg: TrainingConfig
) -> tuple[Predictor, Params]:
    """Fit an untuned LightGBM residual model with fixed, moderate settings."""
    params: Params = {
        "num_leaves": 16,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    estimator = build_lightgbm(params, cfg.seed)
    estimator.fit(features, residuals)
    return TreeModel(estimator, "lightgbm"), params


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors**2)))


def diagnose_origin(
    master: pd.DataFrame, target: str, cfg: TrainingConfig, end: int, horizon: int
) -> dict[str, float | int | str]:
    """Fit the hybrid on the first ``end`` rows and describe its next ``horizon`` days.

    Raises:
        ValueError: if the target is not ``y_fx`` or ``y_cash``.
    """
    if target not in {"y_fx", "y_cash"}:
        raise ValueError("target must be 'y_fx' or 'y_cash'")
    data = master.sort_values("date").reset_index(drop=True)
    history = data.iloc[:end]
    future = data.iloc[end : end + horizon]
    model = fit_hybrid(
        history,
        "y_fx" if target == "y_fx" else "y_cash",
        cfg,
        fit_prophet_base,
        fit_quick_lightgbm,
    )
    if model.base is None:
        raise ValueError("The hybrid model has no Prophet base")
    dates = pd.DatetimeIndex(future["date"])
    features = future_feature_frame(history, dates, model.feature_columns)
    base = model.base.predict(dates)
    tree = model.tree.predict(features)
    actual = future[target].to_numpy(dtype=np.float64)
    last_value = float(history[target].iloc[-1])
    series = history.set_index("date")[target].astype(float)
    residuals = oos_residuals(series, fit_prophet_base, cfg)
    columns = list(model.feature_columns)
    train_features = history.set_index("date").loc[residuals.index, columns]
    first_future = features.iloc[0]
    outside = (first_future < train_features.min()) | (first_future > train_features.max())
    return {
        "origin_rows": end,
        "train_end": str(pd.Timestamp(history["date"].iloc[-1]).date()),
        "last_value": last_value,
        "actual_mean": float(actual.mean()),
        "base_mean": float(base.mean()),
        "tree_mean": float(tree.mean()),
        "bias_base": float((actual - base).mean()),
        "bias_hybrid": float((actual - base - tree).mean()),
        "rmse_random_walk": _rmse(actual - last_value),
        "rmse_base": _rmse(actual - base),
        "rmse_hybrid": _rmse(actual - base - tree),
        "resid_mean_train": float(residuals.mean()),
        "resid_std_train": float(residuals.std()),
        "features_out_of_range": int(outside.sum()),
        "n_features": len(columns),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: print one diagnostic row per backtest origin."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Diagnose the Prophet + LightGBM hybrid.")
    parser.add_argument(
        "--master", type=Path, default=settings.data_dir / "processed" / "master_features.csv"
    )
    parser.add_argument("--target", choices=["y_fx", "y_cash"], default="y_fx")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--min-train", type=int, default=MODEL_MIN_TRAIN_ROWS)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV with the table.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    cfg = load_yaml_model(settings.configs_dir / "training.yaml", TrainingConfig)
    master = read_csv_with_date(args.master)
    origins = fold_origins(len(master), cfg.n_splits, args.horizon, args.min_train)
    rows: list[dict[str, float | int | str]] = []
    for end in origins:
        logger.info("Diagnosing %s with %d training rows", args.target, end)
        rows.append(diagnose_origin(master, args.target, cfg, end, args.horizon))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
