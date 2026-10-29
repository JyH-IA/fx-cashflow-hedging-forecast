"""Tests for data contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from fxhedge.schemas import (
    MACRO_SPEC,
    SALES_SPEC,
    DataContractError,
    required_columns,
    validate_frame,
)


def _sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "occupancy_rate": [0.5, 0.7],
            "sales_usd": [100.0, 200.0],
            "cash_in_usd": [35.0, 70.0],
        }
    )


def test_valid_frame_is_returned_unchanged() -> None:
    frame = _sales()
    assert validate_frame(frame, SALES_SPEC) is frame


def test_missing_column_is_reported() -> None:
    with pytest.raises(DataContractError, match="missing column 'sales_usd'"):
        validate_frame(_sales().drop(columns=["sales_usd"]), SALES_SPEC)


def test_wrong_dtype_is_reported() -> None:
    frame = _sales().assign(sales_usd=["a", "b"])
    with pytest.raises(DataContractError, match="sales_usd"):
        validate_frame(frame, SALES_SPEC)


def test_nulls_ranges_and_duplicates_are_all_reported_together() -> None:
    frame = _sales()
    frame.loc[0, "occupancy_rate"] = 1.5
    frame.loc[1, "sales_usd"] = -1.0
    frame.loc[1, "cash_in_usd"] = None
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(DataContractError) as excinfo:
        validate_frame(frame, SALES_SPEC)
    message = str(excinfo.value)
    assert "above 1.0" in message
    assert "below 0.0" in message
    assert "contains nulls" in message
    assert "duplicates" in message


def test_nullable_columns_accept_missing_values() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    for column in required_columns(MACRO_SPEC):
        if column != "date":
            frame[column] = float("nan")
    assert validate_frame(frame, MACRO_SPEC) is frame


def test_required_columns_lists_the_contract() -> None:
    assert required_columns(SALES_SPEC) == ["date", "occupancy_rate", "sales_usd", "cash_in_usd"]
