"""Typed settings (environment) and YAML-backed configuration models."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class Settings(BaseSettings):
    """Runtime settings read from environment variables or a local .env file.

    Secrets are wrapped in ``SecretStr`` so they never appear in ``repr`` or logs.
    Both secrets are optional so that the project runs on sample data without them.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    banxico_token: SecretStr | None = None
    db_url: SecretStr | None = None
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    configs_dir: Path = Path("configs")
    seed: int = 42


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BusinessConfig(_FrozenModel):
    """Illustrative business assumptions. None of these are real company figures."""

    mxn_cost_share: float = Field(0.82, ge=0.0, le=1.0)
    fixed_hedge_ratio: float = Field(0.5, ge=0.0, le=1.0)
    bank_spread_bps: float = Field(15.0, ge=0.0)
    fixed_fee_usd: float = Field(25.0, ge=0.0)
    rate_mxn_pct: float = Field(10.0, ge=0.0)
    rate_usd_pct: float = Field(4.5, ge=0.0)


class SeriesConfig(_FrozenModel):
    """External series identifiers. Verify every ID against the provider documentation."""

    banxico: dict[str, str]
    yahoo_ticker: str
    fed_funds_fred_id: str


class SimulatorConfig(_FrozenModel):
    """Parameters of the synthetic internal sales generator."""

    start: date
    end: date
    base_daily_sales_usd: float = Field(250_000.0, gt=0.0)
    annual_growth: float = Field(0.04, ge=-0.5, le=1.0)
    seasonality_amplitude: float = Field(0.25, ge=0.0, le=1.0)
    hurricane_penalty: float = Field(0.12, ge=0.0, le=1.0)
    noise_std_pct: float = Field(0.05, ge=0.0, le=1.0)
    cash_conversion_ratio: float = Field(0.35, gt=0.0, le=1.0)


class TrainingConfig(_FrozenModel):
    """Training, validation and feature settings.

    The ``residual_*`` fields control the out-of-sample residuals that the hybrid's tree
    model learns from. With enough rows, the residuals come from blocks of
    ``residual_test_size`` days, close to the forecast horizon. With fewer rows than
    ``residual_min_splits`` such blocks, the older split by ``n_splits`` is used.

    The ``gate_*`` fields control the validation gate. The last ``gate_holdout_fraction`` of
    the residuals is kept for validation; the tree is used only if it lowers the holdout
    error by more than ``gate_min_improvement`` (a fraction) compared with the base model
    alone.
    """

    horizons: list[int] = Field(default_factory=lambda: [30, 60, 90])
    n_splits: int = Field(5, ge=2)
    gap: int = Field(90, ge=0)
    optuna_trials: int = Field(30, ge=1)
    seed: int = 42
    lags: list[int] = Field(default_factory=lambda: [7, 14, 30])
    windows: list[int] = Field(default_factory=lambda: [7, 14, 30])
    interval_level: float = Field(0.95, gt=0.5, lt=1.0)
    residual_test_size: int = Field(90, ge=1)
    residual_min_train: int = Field(365, ge=30)
    residual_min_splits: int = Field(7, ge=2)
    residual_max_splits: int = Field(12, ge=2)
    gate_holdout_fraction: float = Field(0.25, gt=0.0, lt=0.5)
    gate_min_improvement: float = Field(0.0, ge=0.0, lt=1.0)


class MonitoringConfig(_FrozenModel):
    """Thresholds used to decide whether a model must be retrained."""

    psi_bins: int = Field(10, ge=2)
    psi_threshold: float = Field(0.2, gt=0.0)
    rolling_window: int = Field(30, ge=2)
    error_ratio_threshold: float = Field(1.5, gt=1.0)


def load_settings() -> Settings:
    """Build ``Settings`` from the environment and the optional .env file."""
    return Settings()


def load_yaml_model(path: Path, model_cls: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it against ``model_cls``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the content does not match the model.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw: object = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return model_cls.model_validate(raw)
