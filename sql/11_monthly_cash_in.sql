-- Monthly cash-in totals (PostgreSQL and Oracle; SQLite has no EXTRACT).
-- Parameters: :start_date, :end_date.
SELECT
    EXTRACT(YEAR FROM sale_date)  AS year_number,
    EXTRACT(MONTH FROM sale_date) AS month_number,
    SUM(sales_usd)                AS sales_usd,
    SUM(cash_in_usd)              AS cash_in_usd
FROM sales_daily
WHERE sale_date >= :start_date
  AND sale_date <= :end_date
GROUP BY EXTRACT(YEAR FROM sale_date), EXTRACT(MONTH FROM sale_date)
ORDER BY year_number, month_number
