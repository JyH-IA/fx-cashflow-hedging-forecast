"""Static checks of the versioned SQL files."""

from __future__ import annotations

import re
from pathlib import Path

BIND_LIKE = re.compile(r":[A-Za-z_]")
QUERY_FILES = ("10_daily_cash_in.sql", "11_monthly_cash_in.sql")


def _sql_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "sql").rglob("*.sql"))


def _code_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if not line.lstrip().startswith("--")]


def test_sql_files_are_present(repo_root: Path) -> None:
    assert len(_sql_files(repo_root)) >= 4


def test_comments_do_not_contain_bind_like_tokens(repo_root: Path) -> None:
    """SQLAlchemy ``text()`` treats ``:name`` in comments as bind parameters."""
    offenders = []
    for path in _sql_files(repo_root):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("--") and BIND_LIKE.search(line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


def test_query_files_bind_each_date_exactly_once(repo_root: Path) -> None:
    for name in QUERY_FILES:
        code = "\n".join(_code_lines(repo_root / "sql" / name))
        assert code.count(":start_date") == 1
        assert code.count(":end_date") == 1
