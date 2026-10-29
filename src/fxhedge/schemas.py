"""Data contracts for the project tables and a small validator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from pandas.api import types as ptypes
from pydantic import BaseModel, ConfigDict

DType = Literal["datetime", "float", "int", "bool"]


class DataContractError(ValueError):
    """Raised when a DataFrame does not satisfy its data contract."""


class ColumnSpec(BaseModel):
    """Contract of a single column."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: DType
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None


class FrameSpec(BaseModel):
    """Contract of a table: its columns and the column that must be unique."""

    model_config = ConfigDict(frozen=True)

    name: str
    key: str
    columns: tuple[ColumnSpec, ...]


MACRO_SPEC = FrameSpec(
    name="macro_data",
    key="date",
    columns=(
        ColumnSpec(name="date", dtype="datetime"),
        ColumnSpec(name="usdmxn_fix", dtype="float", nullable=True, min_value=0.0),
        ColumnSpec(name="usdmxn_yf", dtype="float", nullable=True, min_value=0.0),
        ColumnSpec(name="cetes_28d", dtype="float", nullable=True, min_value=0.0),
        ColumnSpec(name="fed_funds", dtype="float", nullable=True, min_value=0.0),
        ColumnSpec(name="inpc", dtype="float", nullable=True, min_value=0.0),
        ColumnSpec(name="cancun_arrivals", dtype="float", nullable=True, min_value=0.0),
    ),
)

SALES_SPEC = FrameSpec(
    name="sales_daily",
    key="date",
    columns=(
        ColumnSpec(name="date", dtype="datetime"),
        ColumnSpec(name="occupancy_rate", dtype="float", min_value=0.0, max_value=1.0),
        ColumnSpec(name="sales_usd", dtype="float", min_value=0.0),
        ColumnSpec(name="cash_in_usd", dtype="float", min_value=0.0),
    ),
)

MASTER_SPEC = FrameSpec(
    name="master_features",
    key="date",
    columns=(
        ColumnSpec(name="date", dtype="datetime"),
        ColumnSpec(name="y_fx", dtype="float", min_value=0.0),
        ColumnSpec(name="y_cash", dtype="float", min_value=0.0),
    ),
)


def _dtype_matches(series: pd.Series, dtype: DType) -> bool:
    checks = {
        "datetime": ptypes.is_datetime64_any_dtype,
        "float": ptypes.is_float_dtype,
        "int": ptypes.is_integer_dtype,
        "bool": ptypes.is_bool_dtype,
    }
    return bool(checks[dtype](series))


def _column_errors(df: pd.DataFrame, col: ColumnSpec) -> list[str]:
    if col.name not in df.columns:
        return [f"missing column '{col.name}'"]
    series = df[col.name]
    errors: list[str] = []
    if not _dtype_matches(series, col.dtype):
        errors.append(f"column '{col.name}' has dtype {series.dtype}, expected {col.dtype}")
        return errors
    if not col.nullable and bool(series.isna().any()):
        errors.append(f"column '{col.name}' contains nulls")
    if col.dtype in ("float", "int"):
        if col.min_value is not None and bool((series < col.min_value).any()):
            errors.append(f"column '{col.name}' has values below {col.min_value}")
        if col.max_value is not None and bool((series > col.max_value).any()):
            errors.append(f"column '{col.name}' has values above {col.max_value}")
    return errors


def validate_frame(df: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
    """Validate ``df`` against ``spec`` and return it unchanged.

    Raises:
        DataContractError: listing every violation found.
    """
    errors: list[str] = []
    for col in spec.columns:
        errors.extend(_column_errors(df, col))
    if spec.key in df.columns and bool(df[spec.key].duplicated().any()):
        errors.append(f"key column '{spec.key}' contains duplicates")
    if errors:
        raise DataContractError(f"{spec.name}: " + "; ".join(errors))
    return df


def required_columns(spec: FrameSpec) -> Sequence[str]:
    """Return the column names required by ``spec``."""
    return [col.name for col in spec.columns]
