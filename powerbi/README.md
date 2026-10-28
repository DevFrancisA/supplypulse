# Power BI dashboard

The repo ships the **data model, measures and theme** for the dashboard, not a binary `.pbix`
(a `.pbix` cannot be generated outside Power BI Desktop). Building the report from these files takes about 10 minutes.

## 1. Load the data

`python -m supplypulse.pipeline` writes a ready-made star schema to [`data/`](data/):

| Table | Grain | Used for |
|---|---|---|
| `dim_sku` | one row per SKU | category, supplier, lead time, ABC class, case pack, MOQ |
| `dim_date` | one row per week | year / quarter / month, `period` = train / calibration / test |
| `fact_sales` | SKU x week | units, price, revenue, promo flag |
| `fact_forecast` | SKU x week x model | actual, forecast, absolute error (long format: slice by `model`) |
| `fact_future_forecast` | SKU x horizon week | recursive 6-week forecast |
| `fact_recommendation` | SKU | order-up-to, safety stock, order quantity, status |
| `fact_sim_by_sku` | SKU x policy | simulated fill rate, stockout rate, inventory, cost |
| `fact_sim_weekly` | SKU x policy x week | average on-hand and lost sales over the held-out year |
| `fact_sim_paths` | policy x Monte Carlo path | portfolio KPIs per path (for box plots and intervals) |
| `kpi_summary` | KPI | headline numbers for cards |

In Power BI Desktop: **Get data > Text/CSV**, select all files in `powerbi/data/`, **Load**.

## 2. Relationships (all single-direction, one-to-many)

```
dim_sku[sku_id]     1 -> *  fact_sales[sku_id]
                    1 -> *  fact_forecast[sku_id]
                    1 -> *  fact_future_forecast[sku_id]
                    1 -> *  fact_recommendation[sku_id]
                    1 -> *  fact_sim_by_sku[sku_id]
                    1 -> *  fact_sim_weekly[sku_id]
dim_date[week_idx]  1 -> *  fact_sales[week_idx]
                    1 -> *  fact_forecast[week_idx]
                    1 -> *  fact_sim_weekly[week_idx]
```

`fact_sim_paths` and `kpi_summary` are stand-alone. Set `dim_date[week_start]` to *Date* type and
sort `month_name` by `month`.

## 3. Measures and theme

* Paste [`measures.dax`](measures.dax) into a blank "Measures" table.
* **View > Themes > Browse for themes** and pick [`supplypulse_theme.json`](supplypulse_theme.json).

## 4. Suggested pages

1. **Executive summary** - cards: `WAPE (Test, Gradient boosting)`, `Stockout Reduction %`, `Inventory Reduction %`,
   `Total Cost Reduction %`; clustered column of WAPE by `dim_sku[category]` split by `fact_forecast[model]`
   (filter `fold = test`).
2. **Forecast explorer** - SKU slicer; line chart of `fact_forecast[actual]` and `[forecast]` by `dim_date[week_start]`
   with `model` as legend; `promo_flag` from `fact_sales` as a secondary marker series.
3. **Replenishment** - table from `fact_recommendation` (status, order qty, order value, weeks of cover) with conditional
   formatting on `status`; cards `SKUs to Order`, `Order Value $`, `Safety Stock Value $`.
4. **Simulation** - box plots of `fact_sim_paths[stockout_rate]` and `[avg_inventory_value]` by `policy`;
   line chart of `fact_sim_weekly[avg_lost]` by week and policy.

The interactive what-if analysis lives in the Excel workbook (`outputs/SupplyPulse_WhatIf_Model.xlsx`) and the Streamlit app,
so the dashboard stays a read-only view of pipeline results.
