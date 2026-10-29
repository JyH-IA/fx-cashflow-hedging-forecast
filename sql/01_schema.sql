-- Daily synthetic sales table (PostgreSQL and SQLite compatible).
-- For Oracle use sql/oracle/01_schema.sql. The key column is not named "date"
-- because DATE is a reserved word in Oracle; queries alias it back to "date".
CREATE TABLE IF NOT EXISTS sales_daily (
    sale_date      DATE          NOT NULL PRIMARY KEY,
    occupancy_rate NUMERIC(5, 4) NOT NULL CHECK (occupancy_rate >= 0 AND occupancy_rate <= 1),
    sales_usd      NUMERIC(18, 2) NOT NULL CHECK (sales_usd >= 0),
    cash_in_usd    NUMERIC(18, 2) NOT NULL CHECK (cash_in_usd >= 0)
)
