"""Tests for settings and YAML-backed configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from fxhedge.config import (
    BusinessConfig,
    MonitoringConfig,
    SeriesConfig,
    Settings,
    SimulatorConfig,
    TrainingConfig,
    load_yaml_model,
)

CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "business.yaml": BusinessConfig,
    "series.yaml": SeriesConfig,
    "simulator.yaml": SimulatorConfig,
    "training.yaml": TrainingConfig,
    "monitoring.yaml": MonitoringConfig,
}


@pytest.mark.parametrize("name", sorted(CONFIG_MODELS))
def test_every_shipped_config_is_valid(repo_root: Path, name: str) -> None:
    loaded = load_yaml_model(repo_root / "configs" / name, CONFIG_MODELS[name])
    assert isinstance(loaded, CONFIG_MODELS[name])


def test_business_assumptions_match_the_documented_scenario(repo_root: Path) -> None:
    business = load_yaml_model(repo_root / "configs" / "business.yaml", BusinessConfig)
    assert business.mxn_cost_share == pytest.approx(0.82)


def test_secrets_are_optional_and_hidden() -> None:
    assert Settings(_env_file=None).banxico_token is None
    settings = Settings(_env_file=None, banxico_token="top-secret-value")
    assert "top-secret-value" not in repr(settings)
    assert "top-secret-value" not in str(settings)


def test_load_yaml_model_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml_model(path, BusinessConfig)


@pytest.mark.parametrize(
    "content",
    ["mxn_cost_share: 1.5\n", "unknown_key: 1\n", "bank_spread_bps: -3\n"],
    ids=["out-of-range", "unknown-key", "negative"],
)
def test_load_yaml_model_rejects_invalid_content(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml_model(path, BusinessConfig)


def test_load_yaml_model_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_model(tmp_path / "missing.yaml", BusinessConfig)
