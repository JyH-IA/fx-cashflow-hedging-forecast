"""Small file helpers shared by the pipeline modules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` as CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_csv_with_date(path: Path) -> pd.DataFrame:
    """Read a CSV that has a ``date`` column and parse it as datetime.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, parse_dates=["date"])
