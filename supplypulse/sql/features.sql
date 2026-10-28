-- Leakage-safe feature table.
-- Every window ends at "1 PRECEDING" so a row for week t only sees weeks < t.
-- promo_flag / discount_pct of week t itself are legitimate: promotions are
-- planned ahead of time, so they are known when the forecast is made.
SELECT
    w.sku_id,
    w.week_idx,
    w.week_start,
    w.units,
    w.promo_flag,
    w.discount_pct,
    LAG(w.units, 1)  OVER win AS lag_1,
    LAG(w.units, 2)  OVER win AS lag_2,
    LAG(w.units, 3)  OVER win AS lag_3,
    LAG(w.units, 4)  OVER win AS lag_4,
    LAG(w.units, 8)  OVER win AS lag_8,
    LAG(w.units, 52) OVER win AS lag_52,
    LAG(w.promo_flag, 1) OVER win AS promo_lag_1,
    AVG(w.units) OVER (PARTITION BY w.sku_id ORDER BY w.week_idx
                       ROWS BETWEEN 4  PRECEDING AND 1 PRECEDING) AS ma_4,
    AVG(w.units) OVER (PARTITION BY w.sku_id ORDER BY w.week_idx
                       ROWS BETWEEN 8  PRECEDING AND 1 PRECEDING) AS ma_8,
    AVG(w.units * w.units) OVER (PARTITION BY w.sku_id ORDER BY w.week_idx
                       ROWS BETWEEN 8  PRECEDING AND 1 PRECEDING) AS msq_8,
    AVG(w.units) OVER (PARTITION BY w.sku_id ORDER BY w.week_idx
                       ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS ma_13,
    AVG(CAST(w.promo_flag AS REAL)) OVER (PARTITION BY w.sku_id ORDER BY w.week_idx
                       ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS promo_rate_13,
    -- same-season signal: 3-week average centred on this week last year
    AVG(w.units) OVER (PARTITION BY w.sku_id ORDER BY w.week_idx
                       ROWS BETWEEN 53 PRECEDING AND 51 PRECEDING) AS ly_avg3
FROM weekly_sales AS w
WINDOW win AS (PARTITION BY w.sku_id ORDER BY w.week_idx)
ORDER BY w.sku_id, w.week_idx;
