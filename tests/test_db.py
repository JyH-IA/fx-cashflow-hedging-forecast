"""Tests for database access with a temporary SQLite database (skipped without SQLAlchemy)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

pytest.importorskip("sqlalchemy")

from sqlalchemy.engine import Engine  # noqa: E402

from fxhedge.db import (  # noqa: E402
    apply_schema,
    fetch_daily_cash_in,
    insert_sales,
    make_engine,
    read_sql_file,
)
from fxhedge.schemas import DataContractError  # noqa: E402


def _sales(days: int = 10, start: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=days, freq="D"),
            "occupancy_rate": [0.5 + 0.01 * index for index in range(days)],
            "sales_usd": [1000.0 + index for index in range(days)],
            "cash_in_usd": [350.0 + index for index in range(days)],
        }
    )


@pytest.fixture
def engine(tmp_path: Path, repo_root: Path) -> Engine:
    database = make_engine(SecretStr(f"sqlite+pysqlite:///{tmp_path / 'test.db'}"))
    apply_schema(database, repo_root / "sql" / "01_schema.sql")
    return database


def test_insert_and_query_round_trip(engine: Engine, repo_root: Path) -> None:
    sql = read_sql_file(repo_root / "sql" / "10_daily_cash_in.sql")
    assert insert_sales(engine, _sales()) == 10
    result = fetch_daily_cash_in(engine, sql, date(2024, 1, 3), date(2024, 1, 6))
    assert result["date"].dt.day.tolist() == [3, 4, 5, 6]
    assert result["cash_in_usd"].iloc[0] == pytest.approx(352.0)


def test_reinserting_a_range_replaces_rows(engine: Engine, repo_root: Path) -> None:
    sql = read_sql_file(repo_root / "sql" / "10_daily_cash_in.sql")
    insert_sales(engine, _sales())
    insert_sales(engine, _sales().assign(cash_in_usd=1.0))
    result = fetch_daily_cash_in(engine, sql, date(2024, 1, 1), date(2024, 1, 10))
    assert len(result) == 10
    assert (result["cash_in_usd"] == 1.0).all()


def test_invalid_frames_are_rejected_before_insert(engine: Engine) -> None:
    with pytest.raises(DataContractError):
        insert_sales(engine, _sales().assign(occupancy_rate=2.0))


def test_database_constraints_reject_bad_rows(engine: Engine) -> None:
    with pytest.raises(Exception, match="(?i)check|constraint"):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO sales_daily VALUES ('2024-01-01', 1.5, 1.0, 1.0)"
            )
