# Data contracts

Every table is validated with `validate_frame`. A violation raises `DataContractError` listing all problems.

## `macro_data` (daily)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `date` | datetime | no | Unique key |
| `usdmxn_fix` | float | yes | Banxico FIX, MXN per USD; missing on non-banking days |
| `usdmxn_yf` | float | yes | Yahoo Finance close (`MXN=X`) |
| `cetes_28d` | float | yes | 28-day CETES rate, % per year (Banxico `SF60633`) |
| `fed_funds` | float | yes | Effective federal funds rate, % per year |
| `inpc` | float | yes | Consumer price index; monthly value repeated within the month |
| `cancun_arrivals` | float | yes | International passengers; monthly value repeated within the month |

## `sales_daily` (daily, synthetic)

| Column | Type | Range |
|---|---|---|
| `date` | datetime | unique key |
| `occupancy_rate` | float | 0 to 1 |
| `sales_usd` | float | at least 0 |
| `cash_in_usd` | float | at least 0 |

In SQL the key column is `sale_date` (DATE is reserved in Oracle); queries alias it back to `date`.

## `master_features`

- `date`, targets `y_fx` (USD/MXN) and `y_cash` (daily cash-in in USD).
- Exogenous variables that have data, and their lags (7, 14, 30 days) and rolling mean/std (7, 14, 30 days) on shifted series.
- Calendar columns: `day_of_week`, `month`, `quarter`, `is_weekend`, `is_holiday_mx`, `is_high_season`, `is_hurricane_season`.
- Same-day columns that would leak the target (`usdmxn_fix`, `usdmxn_yf`, `sales_usd`, `cash_in_usd`, `occupancy_rate`) are dropped.

## `data_source.txt`

A one-word text file in `data/raw/` (written by `make sample` and `make ingest`) and in `data/processed/` (copied by `make features`). Its value is `synthetic` or `banxico`; a missing or unrecognised file reads as `unknown`. The dashboard reads the copy in `data/processed/`.

## API response (`GET /forecast`)

`run_date`, `horizon` (30, 60 or 90), `scenario` (`base`, `alza`, `baja`), `disclaimer` and `points`, each with `date`, `fx_predicted`, `cash_expected_usd`, `fx_lower`, `fx_upper`, `cash_lower`, `cash_upper`.
