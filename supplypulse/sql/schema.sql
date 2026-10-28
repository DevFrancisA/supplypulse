DROP TABLE IF EXISTS weekly_sales;
DROP TABLE IF EXISTS skus;

CREATE TABLE skus (
    sku_id          TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    supplier        TEXT NOT NULL,
    unit_cost       REAL NOT NULL,
    unit_price      REAL NOT NULL,
    lead_time_weeks INTEGER NOT NULL,
    case_pack       INTEGER NOT NULL,
    moq             INTEGER NOT NULL
);

CREATE TABLE weekly_sales (
    sku_id       TEXT    NOT NULL REFERENCES skus (sku_id),
    week_idx     INTEGER NOT NULL,
    week_start   TEXT    NOT NULL,
    units        REAL    NOT NULL,
    price        REAL    NOT NULL,
    promo_flag   INTEGER NOT NULL,
    discount_pct REAL    NOT NULL,
    PRIMARY KEY (sku_id, week_idx)
);

CREATE INDEX idx_sales_week ON weekly_sales (week_idx);
