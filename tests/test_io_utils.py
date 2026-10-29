"""Tests for file helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from fxhedge.io_utils import read_csv_with_date, sha256_file, write_csv


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256_file(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_csv_round_trip_creates_parent_directories(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "value": [1.5]})
    path = tmp_path / "nested" / "dir" / "file.csv"
    write_csv(frame, path)
    loaded = read_csv_with_date(path)
    assert loaded.equals(frame)


def test_reading_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_csv_with_date(tmp_path / "missing.csv")
