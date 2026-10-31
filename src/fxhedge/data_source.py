"""Marker file that records whether the raw data is synthetic or comes from Banxico.

The dashboard reads it to tell the reader what kind of data stands behind the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DataSource = Literal["synthetic", "banxico", "unknown"]

DATA_SOURCE_FILE = "data_source.txt"
_RECORDED: tuple[DataSource, ...] = ("synthetic", "banxico")


def write_data_source(directory: Path, source: DataSource) -> Path:
    """Write the marker file in ``directory`` (created if needed) and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATA_SOURCE_FILE
    path.write_text(f"{source}\n", encoding="utf-8")
    return path


def read_data_source(directory: Path) -> DataSource:
    """Source recorded in ``directory``, or ``unknown`` if the file is missing or unreadable."""
    path = directory / DATA_SOURCE_FILE
    if not path.is_file():
        return "unknown"
    text = path.read_text(encoding="utf-8").strip()
    for source in _RECORDED:
        if text == source:
            return source
    return "unknown"
