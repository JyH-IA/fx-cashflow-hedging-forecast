-- Daily cash-in series between two dates (inclusive). Parameters: :start_date, :end_date.
SELECT
    sale_date      AS "date",
    occupancy_rate AS occupancy_rate,
    sales_usd      AS sales_usd,
    cash_in_usd    AS cash_in_usd
FROM sales_daily
WHERE sale_date >= :start_date
  AND sale_date <= :end_date
ORDER BY sale_date
