"""Database access: parameterized queries only, via SQLAlchemy 2.0."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import SecretStr
from sqlalchemy import Date, bindparam, create_engine, text
from sqlalchemy.engine import Engine

from fxhedge.schemas import SALES_SPEC, validate_frame

_INSERT_SALES_SQL = (
    "INSERT INTO sales_daily (sale_date, occupancy_rate, sales_usd, cash_in_usd) "
    "VALUES (:sale_date, :occupancy_rate, :sales_usd, :cash_in_usd)"
)
_DELETE_RANGE_SQL = (
    "DELETE FROM sales_daily WHERE sale_date >= :start_date AND sale_date <= :end_date"
)


def make_engine(db_url: SecretStr) -> Engine:
    """Create an engine from a secret URL. The URL is never logged."""
    return create_engine(db_url.get_secret_value(), pool_pre_ping=True)


def read_sql_file(path: Path) -> str:
    """Read a versioned SQL file as text.

    Keep ``:name`` tokens out of SQL comments: SQLAlchemy ``text()`` treats them as binds.
    """
    return path.read_text(encoding="utf-8")


def apply_schema(engine: Engine, schema_path: Path) -> None:
    """Execute a schema file that contains a single statement."""
    statement = read_sql_file(schema_path)
    with engine.begin() as conn:
        conn.exec_driver_sql(statement)


def fetch_daily_cash_in(engine: Engine, sql: str, start: date, end: date) -> pd.DataFrame:
    """Run a query with :start_date and :end_date parameters and validate the result."""
    stmt = text(sql).bindparams(
        bindparam("start_date", value=start, type_=Date),
        bindparam("end_date", value=end, type_=Date),
    )
    with engine.connect() as conn:
        result = conn.execute(stmt)
        frame = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("occupancy_rate", "sales_usd", "cash_in_usd"):
        frame[column] = frame[column].astype(float)
    return validate_frame(frame, SALES_SPEC)


def insert_sales(engine: Engine, frame: pd.DataFrame) -> int:
    """Replace the date range covered by ``frame`` and insert its rows.

    Returns:
        The number of inserted rows.

    Raises:
        DataContractError: if ``frame`` does not satisfy the sales contract.
    """
    validate_frame(frame, SALES_SPEC)
    days = pd.DatetimeIndex(frame["date"]).date
    records = [
        {
            "sale_date": day,
            "occupancy_rate": float(occupancy),
            "sales_usd": float(sales),
            "cash_in_usd": float(cash_in),
        }
        for day, occupancy, sales, cash_in in zip(
            days,
            frame["occupancy_rate"],
            frame["sales_usd"],
            frame["cash_in_usd"],
            strict=True,
        )
    ]
    if not records:
        return 0
    delete_stmt = text(_DELETE_RANGE_SQL).bindparams(
        bindparam("start_date", type_=Date), bindparam("end_date", type_=Date)
    )
    insert_stmt = text(_INSERT_SALES_SQL).bindparams(bindparam("sale_date", type_=Date))
    with engine.begin() as conn:
        conn.execute(delete_stmt, {"start_date": min(days), "end_date": max(days)})
        conn.execute(insert_stmt, records)
    return len(records)
