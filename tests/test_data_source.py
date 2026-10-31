"""Tests for the marker file that records where the raw data comes from."""

from __future__ import annotations

from pathlib import Path

import pytest

from fxhedge.data_source import (
    DATA_SOURCE_FILE,
    DataSource,
    read_data_source,
    write_data_source,
)


def test_a_missing_marker_reads_as_unknown(tmp_path: Path) -> None:
    assert read_data_source(tmp_path) == "unknown"


@pytest.mark.parametrize("source", ["synthetic", "banxico"])
def test_the_marker_round_trips(tmp_path: Path, source: DataSource) -> None:
    write_data_source(tmp_path, source)
    assert read_data_source(tmp_path) == source


def test_writing_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "nested"
    path = write_data_source(target, "synthetic")
    assert path == target / DATA_SOURCE_FILE
    assert path.is_file()


def test_a_later_write_replaces_the_earlier_one(tmp_path: Path) -> None:
    write_data_source(tmp_path, "synthetic")
    write_data_source(tmp_path, "banxico")
    assert read_data_source(tmp_path) == "banxico"


def test_an_unrecognised_marker_reads_as_unknown(tmp_path: Path) -> None:
    (tmp_path / DATA_SOURCE_FILE).write_text("something else\n", encoding="utf-8")
    assert read_data_source(tmp_path) == "unknown"
