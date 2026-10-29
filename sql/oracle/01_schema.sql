-- Oracle variant of the daily synthetic sales table (not verified against a live Oracle instance).
CREATE TABLE sales_daily (
    sale_date      DATE           NOT NULL PRIMARY KEY,
    occupancy_rate NUMBER(5, 4)   NOT NULL CHECK (occupancy_rate >= 0 AND occupancy_rate <= 1),
    sales_usd      NUMBER(18, 2)  NOT NULL CHECK (sales_usd >= 0),
    cash_in_usd    NUMBER(18, 2)  NOT NULL CHECK (cash_in_usd >= 0)
)
