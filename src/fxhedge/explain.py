"""Model interpretability: SHAP values, top drivers and a summary figure."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fxhedge.config import load_settings
from fxhedge.io_utils import read_csv_with_date
from fxhedge.model_training import TARGETS, TrainedModel, load_trained_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapResult:
    """SHAP values with shape ``(n_samples, n_features)`` and their feature names."""

    values: NDArray[np.float64]
    feature_names: tuple[str, ...]
    base_value: float


def compute_shap_values(model: TrainedModel, features: pd.DataFrame) -> ShapResult:
    """Compute SHAP values for the tree part of ``model``.

    Raises:
        TypeError: if the tree model does not expose its estimator.
    """
    import shap

    estimator = getattr(model.tree, "estimator", None)
    if estimator is None:
        raise TypeError("The tree model does not expose an estimator for SHAP")
    table = features.loc[:, list(model.feature_columns)]
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(table), dtype=np.float64)
    base_value = float(np.ravel(explainer.expected_value)[0])
    return ShapResult(values=values, feature_names=tuple(table.columns), base_value=base_value)


def mean_abs_shap(result: ShapResult) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted from most to least influential."""
    if result.values.ndim != 2 or result.values.shape[1] != len(result.feature_names):
        raise ValueError("SHAP values must be 2-D with one column per feature")
    importance = np.abs(result.values).mean(axis=0)
    return pd.Series(importance, index=list(result.feature_names)).sort_values(ascending=False)


def top_drivers(result: ShapResult, k: int = 3) -> list[str]:
    """Names of the ``k`` most influential features."""
    if k < 1:
        raise ValueError("k must be positive")
    return list(mean_abs_shap(result).index[:k])


def describe_drivers(result: ShapResult, k: int = 3) -> str:
    """One sentence with the top drivers and their share of the total SHAP impact."""
    importance = mean_abs_shap(result)
    total = float(importance.sum())
    if total == 0.0:
        return "The residual model had no measurable impact on the forecast."
    parts = [f"{name} ({value / total:.0%})" for name, value in importance.head(k).items()]
    return "Main drivers of the residual model: " + ", ".join(parts) + "."


def save_shap_summary(result: ShapResult, out_path: Path, max_features: int = 15) -> None:
    """Save a horizontal bar chart of mean absolute SHAP values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance = mean_abs_shap(result).head(max_features).iloc[::-1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 0.4 * len(importance) + 1.5))
    axis.barh(list(importance.index), importance.to_numpy())
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title("Feature influence on the residual model")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: SHAP figures and a driver summary per target."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Explain trained models with SHAP.")
    parser.add_argument(
        "--master", type=Path, default=settings.data_dir / "processed" / "master_features.csv"
    )
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--rows", type=int, default=365, help="Most recent rows to explain.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    master = read_csv_with_date(args.master).set_index("date").tail(args.rows)
    lines = ["# Model drivers (SHAP)", ""]
    for target in TARGETS:
        model = load_trained_model(args.models_dir / target)
        result = compute_shap_values(model, master)
        save_shap_summary(result, args.out / "figures" / f"shap_summary_{target}.png")
        lines.append(f"- **{target}**: {describe_drivers(result)}")
    (args.out / "shap_drivers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote SHAP outputs to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
