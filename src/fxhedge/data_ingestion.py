"""Data ingestion: Banxico SIE, Yahoo Finance, manual CSVs and synthetic sales.

Secrets are only read from ``Settings`` and are never logged. The Banxico token is
sent in the ``Bmx-Token`` header, never in the URL.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests
from pydantic import SecretStr

from fxhedge.config import (
    SeriesConfig,
    Settings,
    SimulatorConfig,
    load_settings,
    load_yaml_model,
)
from fxhedge.data_source import write_data_source
from fxhedge.io_utils import read_csv_with_date, write_csv
from fxhedge.schemas import MACRO_SPEC, SALES_SPEC, validate_frame
from fxhedge.simulator import simulate_internal_sales

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

BANXICO_URL = (
    "https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
    "{series_id}/datos/{start}/{end}"
)
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MONTHLY_COLUMNS = ("inpc", "cancun_arrivals")


class IngestionError(RuntimeError):
    """Raised when external data cannot be retrieved or parsed."""


class _RetryableError(Exception):
    """Internal marker for responses worth retrying."""


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(date(cursor.year, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def default_end(today: date | None = None) -> date:
    """Last day to ingest when ``--end`` is not given: yesterday, so its FIX is published."""
    return (today or date.today()) - timedelta(days=1)


def _get_json(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    timeout_s: float,
    max_retries: int,
    sleep: Callable[[float], None],
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=timeout_s)
            if response.status_code in _RETRYABLE_STATUS:
                raise _RetryableError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, _RetryableError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                sleep(0.5 * (2**attempt))
    raise IngestionError(f"Banxico request failed after {max_retries} attempts") from last_error


def parse_banxico_response(payload: object) -> pd.DataFrame:
    """Parse a Banxico SIE JSON payload into a long DataFrame.

    Returns:
        Columns ``date``, ``series_id`` and ``value``. Observations that are not
        numeric (for example ``N/E``) are skipped.

    Raises:
        IngestionError: if the payload does not have the expected structure.
    """
    if not isinstance(payload, dict):
        raise IngestionError("Unexpected Banxico payload: not a JSON object")
    bmx = payload.get("bmx")
    series_list = bmx.get("series") if isinstance(bmx, dict) else None
    if not isinstance(series_list, list):
        raise IngestionError("Unexpected Banxico payload: missing 'bmx.series'")
    rows: list[dict[str, object]] = []
    for item in series_list:
        if not isinstance(item, dict):
            raise IngestionError("Unexpected Banxico payload: invalid series entry")
        series_id = item.get("idSerie")
        observations = item.get("datos", [])
        if not isinstance(series_id, str) or not isinstance(observations, list):
            raise IngestionError("Unexpected Banxico payload: invalid series fields")
        for obs in observations:
            if not isinstance(obs, dict):
                raise IngestionError("Unexpected Banxico payload: invalid observation")
            try:
                value = float(str(obs.get("dato", "")).replace(",", ""))
            except ValueError:
                continue
            rows.append(
                {
                    "date": pd.to_datetime(str(obs.get("fecha")), format="%d/%m/%Y"),
                    "series_id": series_id,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=["date", "series_id", "value"])


def fetch_banxico_series(
    series_ids: Sequence[str],
    start: date,
    end: date,
    token: SecretStr,
    *,
    timeout_s: float = 20.0,
    max_retries: int = 3,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download Banxico SIE series, one request per series and calendar year.

    Returns:
        A DataFrame with a ``date`` column and one column per series ID.
    """
    http = session if session is not None else requests.Session()
    headers = {"Bmx-Token": token.get_secret_value(), "Accept": "application/json"}
    frames: list[pd.DataFrame] = []
    for series_id in series_ids:
        for chunk_start, chunk_end in _year_chunks(start, end):
            url = BANXICO_URL.format(
                series_id=series_id, start=chunk_start.isoformat(), end=chunk_end.isoformat()
            )
            payload = _get_json(http, url, headers, timeout_s, max_retries, sleep)
            frames.append(parse_banxico_response(payload))
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "series_id", "value"]
    )
    if long.empty:
        return pd.DataFrame({"date": pd.to_datetime([])})
    long["date"] = pd.to_datetime(long["date"])
    wide = long.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    wide.columns.name = None
    flattened: pd.DataFrame = wide.reset_index()
    return flattened


def fetch_yahoo_fx(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download the daily close of ``ticker`` from Yahoo Finance as ``usdmxn_yf``."""
    import yfinance as yf

    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if data is None or data.empty:
        raise IngestionError(f"Yahoo Finance returned no data for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    frame: pd.DataFrame = close.rename("usdmxn_yf").reset_index()
    frame.columns = ["date", "usdmxn_yf"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["usdmxn_yf"] = frame["usdmxn_yf"].astype(float)
    return frame


def load_fed_funds_csv(path: Path) -> pd.DataFrame:
    """Load a FRED-style CSV (DATE/observation_date and DFF/fed_funds columns)."""
    if not path.exists():
        raise FileNotFoundError(f"Fed funds CSV not found: {path}")
    raw = pd.read_csv(path).rename(
        columns={"DATE": "date", "observation_date": "date", "DFF": "fed_funds"}
    )
    if not {"date", "fed_funds"}.issubset(raw.columns):
        raise IngestionError("Fed funds CSV must have date and fed_funds (or DFF) columns")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            "fed_funds": pd.to_numeric(raw["fed_funds"], errors="coerce"),
        }
    )
    return frame.dropna(subset=["fed_funds"])


def load_sectur_csv(path: Path) -> pd.DataFrame:
    """Load monthly Cancun arrivals from a manually downloaded CSV.

    The file must have columns ``month`` (YYYY-MM) and ``cancun_arrivals``.
    """
    if not path.exists():
        raise FileNotFoundError(f"SECTUR CSV not found: {path} (manual download, see README)")
    raw = pd.read_csv(path)
    if not {"month", "cancun_arrivals"}.issubset(raw.columns):
        raise IngestionError("SECTUR CSV must have month and cancun_arrivals columns")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw["month"], format="%Y-%m"),
            "cancun_arrivals": pd.to_numeric(raw["cancun_arrivals"], errors="coerce"),
        }
    )


def load_sales(
    engine: Engine | None,
    sample_path: Path,
    start: date,
    end: date,
    sql_path: Path = Path("sql/10_daily_cash_in.sql"),
) -> pd.DataFrame:
    """Load daily sales from SQL when an engine is given, otherwise from a CSV sample."""
    if engine is not None:
        from fxhedge.db import fetch_daily_cash_in, read_sql_file

        return fetch_daily_cash_in(engine, read_sql_file(sql_path), start, end)
    frame = read_csv_with_date(sample_path)
    mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    return validate_frame(frame.loc[mask].reset_index(drop=True), SALES_SPEC)


def build_macro_dataset(
    settings: Settings,
    start: date,
    end: date,
    *,
    series_cfg: SeriesConfig | None = None,
    fed_funds_path: Path | None = None,
    sectur_path: Path | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Build the daily macro table from Banxico, Yahoo Finance and manual CSVs.

    Raises:
        IngestionError: if the Banxico token is not configured.
    """
    if settings.banxico_token is None:
        raise IngestionError("BANXICO_TOKEN is not set; use data/sample/ to run without it")
    cfg = series_cfg or load_yaml_model(settings.configs_dir / "series.yaml", SeriesConfig)
    external = settings.data_dir / "external"
    fed_path = fed_funds_path or external / "fed_funds.csv"
    sectur = sectur_path or external / "sectur_arribos.csv"

    calendar = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
    id_to_name = {series_id: name for name, series_id in cfg.banxico.items()}
    banxico = fetch_banxico_series(
        list(id_to_name), start, end, settings.banxico_token, session=session
    ).rename(columns=id_to_name)

    macro = calendar.merge(banxico, on="date", how="left")
    macro = macro.merge(fetch_yahoo_fx(cfg.yahoo_ticker, start, end), on="date", how="left")
    for path, loader in ((fed_path, load_fed_funds_csv), (sectur, load_sectur_csv)):
        if path.exists():
            macro = macro.merge(loader(path), on="date", how="left")
        else:
            logger.warning("Optional file missing, column left empty: %s", path)

    for column in MACRO_SPEC.columns:
        if column.name not in macro.columns:
            macro[column.name] = float("nan")
    for name in _MONTHLY_COLUMNS:
        macro[name] = macro[name].ffill()
    ordered = macro[[column.name for column in MACRO_SPEC.columns]].copy()
    for column in MACRO_SPEC.columns:
        if column.dtype == "float":
            ordered[column.name] = ordered[column.name].astype(float)
    return validate_frame(ordered, MACRO_SPEC)


def _store_sales_in_db(settings: Settings, sales: pd.DataFrame) -> None:
    if settings.db_url is None:
        return
    from fxhedge.db import apply_schema, insert_sales, make_engine

    engine = make_engine(settings.db_url)
    apply_schema(engine, Path("sql/01_schema.sql"))
    inserted = insert_sales(engine, sales)
    logger.info("Inserted %d synthetic sales rows into the database", inserted)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: simulate sales and, if possible, fetch macro data.

    The start date comes from ``configs/simulator.yaml``. The end date is ``--end`` or, when
    it is not given, yesterday.
    """
    parser = argparse.ArgumentParser(description="Ingest macro data and simulate sales.")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--skip-macro", action="store_true", help="Only simulate sales.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    sim_cfg = load_yaml_model(settings.configs_dir / "simulator.yaml", SimulatorConfig)
    start = args.start or sim_cfg.start
    end = args.end or default_end()
    raw_dir = settings.data_dir / "raw"

    sales = simulate_internal_sales(sim_cfg, start, end, settings.seed)
    write_csv(sales, raw_dir / "internal_sales_simulated.csv")
    _store_sales_in_db(settings, sales)
    if args.skip_macro:
        return 0
    try:
        macro = build_macro_dataset(settings, start, end)
    except IngestionError as exc:
        logger.error("Macro ingestion failed: %s", exc)
        return 1
    write_csv(macro, raw_dir / "macro_data.csv")
    write_data_source(raw_dir, "banxico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
