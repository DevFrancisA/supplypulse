-- ABC classification on trailing-52-week revenue (Pareto: A = first 80%, B = next 15%, C = rest).
WITH recent AS (
    SELECT sku_id, SUM(units * price) AS revenue
    FROM weekly_sales
    WHERE week_idx > (SELECT MAX(week_idx) FROM weekly_sales) - 52
    GROUP BY sku_id
),
ranked AS (
    SELECT sku_id,
           revenue,
           SUM(revenue) OVER (ORDER BY revenue DESC, sku_id
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
             / SUM(revenue) OVER ()                       AS cum_share,
           revenue / SUM(revenue) OVER ()                 AS share
    FROM recent
)
SELECT sku_id,
       revenue,
       cum_share,
       CASE WHEN cum_share - share < 0.80 THEN 'A'
            WHEN cum_share - share < 0.95 THEN 'B'
            ELSE 'C' END AS abc_class
FROM ranked
ORDER BY revenue DESC;
