"""Tests for data ingestion with fake HTTP sessions (no network access)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
from pydantic import SecretStr

from fxhedge import data_ingestion as ingestion
from fxhedge.config import SeriesConfig, Settings
from fxhedge.data_ingestion import (
    IngestionError,
    build_macro_dataset,
    default_end,
    fetch_banxico_series,
    load_fed_funds_csv,
    load_sales,
    load_sectur_csv,
    parse_banxico_response,
)
from fxhedge.io_utils import write_csv

TOKEN = SecretStr("unit-test-token")


class FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def _payload(series_id: str, values: dict[str, str]) -> dict[str, Any]:
    observations = [{"fecha": day, "dato": value} for day, value in values.items()]
    return {"bmx": {"series": [{"idSerie": series_id, "datos": observations}]}}


class RecordingSession:
    """Fails once with HTTP 503, then answers every request."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self._fail_first = fail_first

    def get(self, url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append((url, headers))
        if self._fail_first and len(self.calls) == 1:
            return FakeResponse(503, {})
        series_id = url.split("/series/")[1].split("/")[0]
        return FakeResponse(200, _payload(series_id, {"02/01/2024": "17.5", "03/01/2024": "17.6"}))


class AlwaysFailing:
    def get(self, url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        return FakeResponse(500, {})


def _no_sleep(_: float) -> None:
    return None


def test_parse_skips_non_numeric_observations() -> None:
    payload = _payload("SF1", {"02/01/2024": "17.1", "03/01/2024": "N/E", "04/01/2024": "1,234.5"})
    table = parse_banxico_response(payload)
    assert table["value"].tolist() == [17.1, 1234.5]
    assert table["date"].iloc[0] == pd.Timestamp("2024-01-02")


@pytest.mark.parametrize(
    "payload",
    [[], {"x": 1}, {"bmx": {"series": "no"}}, {"bmx": {"series": [1]}}],
    ids=["list", "no-bmx", "series-not-list", "entry-not-dict"],
)
def test_parse_rejects_unexpected_structures(payload: object) -> None:
    with pytest.raises(IngestionError):
        parse_banxico_response(payload)


def test_fetch_retries_after_503_and_keeps_the_token_out_of_the_url() -> None:
    session = RecordingSession(fail_first=True)
    table = fetch_banxico_series(
        ["SF1", "SP1"],
        date(2024, 1, 1),
        date(2024, 1, 31),
        TOKEN,
        session=session,  # type: ignore[arg-type]
        sleep=_no_sleep,
    )
    assert len(session.calls) == 3
    assert all("unit-test-token" not in url for url, _ in session.calls)
    assert session.calls[1][1]["Bmx-Token"] == "unit-test-token"
    assert set(table.columns) == {"date", "SF1", "SP1"}


def test_fetch_splits_requests_by_calendar_year() -> None:
    session = RecordingSession()
    fetch_banxico_series(
        ["SF1"],
        date(2023, 6, 1),
        date(2024, 2, 1),
        TOKEN,
        session=session,  # type: ignore[arg-type]
        sleep=_no_sleep,
    )
    assert len(session.calls) == 2
    assert "2023-06-01/2023-12-31" in session.calls[0][0]
    assert "2024-01-01/2024-02-01" in session.calls[1][0]


def test_fetch_gives_up_after_the_configured_attempts() -> None:
    with pytest.raises(IngestionError, match="after 3 attempts"):
        fetch_banxico_series(
            ["SF1"],
            date(2024, 1, 1),
            date(2024, 1, 31),
            TOKEN,
            session=AlwaysFailing(),  # type: ignore[arg-type]
            sleep=_no_sleep,
        )


def test_loaders_read_and_validate_manual_csvs(tmp_path: Path) -> None:
    pd.DataFrame({"observation_date": ["2024-01-01"], "DFF": [5.33]}).to_csv(
        tmp_path / "fed.csv", index=False
    )
    pd.DataFrame({"month": ["2024-01"], "cancun_arrivals": [1_000_000]}).to_csv(
        tmp_path / "sectur.csv", index=False
    )
    assert load_fed_funds_csv(tmp_path / "fed.csv")["fed_funds"].iloc[0] == pytest.approx(5.33)
    assert load_sectur_csv(tmp_path / "sectur.csv")["date"].iloc[0] == pd.Timestamp("2024-01-01")
    pd.DataFrame({"wrong": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(IngestionError):
        load_sectur_csv(tmp_path / "bad.csv")
    with pytest.raises(IngestionError):
        load_fed_funds_csv(tmp_path / "bad.csv")
    with pytest.raises(FileNotFoundError):
        load_sectur_csv(tmp_path / "missing.csv")


def test_build_macro_dataset_joins_sources_and_fills_monthly_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_yahoo(ticker: str, start: date, end: date) -> pd.DataFrame:
        days = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame({"date": days, "usdmxn_yf": [17.4, 17.5]})

    monkeypatch.setattr(ingestion, "fetch_yahoo_fx", fake_yahoo)
    external = tmp_path / "external"
    external.mkdir()
    pd.DataFrame({"month": ["2024-01"], "cancun_arrivals": [1_000_000]}).to_csv(
        external / "sectur_arribos.csv", index=False
    )
    series = SeriesConfig(
        banxico={"usdmxn_fix": "SF43718", "inpc": "SP1"},
        yahoo_ticker="MXN=X",
        fed_funds_fred_id="DFF",
    )
    settings = Settings(_env_file=None, banxico_token=TOKEN, data_dir=tmp_path)
    macro = build_macro_dataset(
        settings,
        date(2024, 1, 1),
        date(2024, 1, 5),
        series_cfg=series,
        session=RecordingSession(),  # type: ignore[arg-type]
    )
    assert len(macro) == 5
    assert macro["usdmxn_fix"].notna().sum() == 2
    assert macro["inpc"].iloc[-1] == pytest.approx(17.6)
    assert (macro["cancun_arrivals"] == 1_000_000).all()
    assert macro["fed_funds"].isna().all()


def test_build_macro_dataset_requires_a_token() -> None:
    with pytest.raises(IngestionError, match="BANXICO_TOKEN"):
        build_macro_dataset(Settings(_env_file=None), date(2024, 1, 1), date(2024, 1, 5))


def test_default_end_is_yesterday() -> None:
    assert default_end(date(2026, 10, 31)) == date(2026, 10, 30)
    assert default_end(date(2026, 1, 1)) == date(2025, 12, 31)


def test_load_sales_filters_the_sample_file(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    sample = pd.DataFrame(
        {
            "date": dates,
            "occupancy_rate": 0.6,
            "sales_usd": 100.0,
            "cash_in_usd": 35.0,
        }
    )
    write_csv(sample, tmp_path / "sales.csv")
    loaded = load_sales(None, tmp_path / "sales.csv", date(2024, 1, 3), date(2024, 1, 5))
    assert loaded["date"].dt.day.tolist() == [3, 4, 5]


def test_cli_without_macro_writes_the_simulated_sales(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONFIGS_DIR", str(repo_root / "configs"))
    code = ingestion.main(["--skip-macro", "--start", "2024-01-01", "--end", "2024-03-31"])
    assert code == 0
    written = pd.read_csv(tmp_path / "data" / "raw" / "internal_sales_simulated.csv")
    assert len(written) == 91


def test_cli_without_token_fails_cleanly_after_writing_sales(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONFIGS_DIR", str(repo_root / "configs"))
    assert ingestion.main(["--start", "2024-01-01", "--end", "2024-01-31"]) == 1
    assert (tmp_path / "data" / "raw" / "internal_sales_simulated.csv").exists()
    assert not (tmp_path / "data" / "raw" / "macro_data.csv").exists()
