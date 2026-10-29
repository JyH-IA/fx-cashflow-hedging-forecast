"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent from the developer's environment variables."""
    for name in ("BANXICO_TOKEN", "DB_URL", "DATA_DIR", "MODELS_DIR", "CONFIGS_DIR", "SEED"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo_root() -> Path:
    """Root directory of the repository."""
    return Path(__file__).resolve().parent.parent
