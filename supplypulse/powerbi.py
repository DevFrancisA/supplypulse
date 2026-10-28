"""Star-schema CSV exports for the Power BI dashboard (see powerbi/README.md)."""
from __future__ import annotations

import json

import pandas as pd

from .config import POWERBI_DIR


def export(skus, sales, preds, future, rec, sim_by_sku, sim_weekly, sim_paths, metrics, cfg) -> None:
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    dim_sku = skus.drop(columns=["revenue"], errors="ignore")
    dim_sku.to_csv(POWERBI_DIR / "dim_sku.csv", index=False)

    weeks = (sales[["week_idx", "week_start"]].drop_duplicates().sort_values("week_idx"))
    d = pd.to_datetime(weeks["week_start"])
    dim_date = weeks.assign(
        week_start=d.dt.strftime("%Y-%m-%d"), year=d.dt.year, quarter=d.dt.quarter,
        month=d.dt.month, month_name=d.dt.strftime("%b"),
        iso_week=d.dt.isocalendar().week.astype(int),
        period=["train" if i < cfg.calib_start else "calibration" if i < cfg.test_start else "test"
                for i in weeks.week_idx],
    )
    dim_date.to_csv(POWERBI_DIR / "dim_date.csv", index=False)

    fact_sales = sales.assign(revenue=(sales.units * sales.price).round(2))
    fact_sales.to_csv(POWERBI_DIR / "fact_sales.csv", index=False)

    p = preds.copy()
    long = p.melt(id_vars=["sku_id", "week_idx", "fold", "actual"],
                  value_vars=["fc_gbm", "fc_ma4", "fc_snaive"], var_name="model", value_name="forecast")
    long["model"] = long.model.map({"fc_gbm": "Gradient boosting", "fc_ma4": "4-wk moving average",
                                    "fc_snaive": "Seasonal naive"})
    long = long.dropna(subset=["forecast"])
    long["abs_error"] = (long.actual - long.forecast).abs()
    long.round(3).to_csv(POWERBI_DIR / "fact_forecast.csv", index=False)

    future.round(3).to_csv(POWERBI_DIR / "fact_future_forecast.csv", index=False)
    rec.to_csv(POWERBI_DIR / "fact_recommendation.csv", index=False)
    sim_by_sku.round(5).to_csv(POWERBI_DIR / "fact_sim_by_sku.csv", index=False)
    sim_weekly.round(3).to_csv(POWERBI_DIR / "fact_sim_weekly.csv", index=False)
    sim_paths.round(4).to_csv(POWERBI_DIR / "fact_sim_paths.csv", index=False)

    sm = metrics["simulation"]
    kpi = pd.DataFrame([
        ("WAPE - moving average baseline", metrics["forecast"]["wape_ma4"]),
        ("WAPE - gradient boosting", metrics["forecast"]["wape_gbm"]),
        ("Stockout reduction (SKU-weeks)", sm["stockout_reduction_pct"] / 100),
        ("Lost-units reduction", sm["lost_units_reduction_pct"] / 100),
        ("Average inventory reduction", sm["inventory_reduction_pct"] / 100),
        ("Fill rate - baseline", sm["baseline"]["fill_rate"]),
        ("Fill rate - SupplyPulse", sm["supplypulse"]["fill_rate"]),
    ], columns=["kpi", "value"])
    kpi.to_csv(POWERBI_DIR / "kpi_summary.csv", index=False)
    (POWERBI_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
