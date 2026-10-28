"""Closed-form what-if engine. Mirrors the formulas in the Excel workbook so the
Streamlit app and the spreadsheet always agree."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .optimize import normal_loss


def scenario(rec: pd.DataFrame, optimised: bool = True, service_level: float = 0.95,
             uplift: float = 0.0, delay_weeks: int = 0, holding_rate: float = 0.25) -> pd.DataFrame:
    """Recompute order-up-to levels and order quantities under a what-if scenario.

    optimised     -- use the PuLP per-SKU z; otherwise one global `service_level` for all SKUs
    uplift        -- demand change vs. forecast (0.10 = +10%), scales mean and sigma
    delay_weeks   -- extra supplier lead time; sigma scales by sqrt((L'+1)/(L+1))
    """
    l0 = rec.lead_time_weeks.to_numpy(float)
    lead = l0 + delay_weeks
    weekly = rec.cover_demand.to_numpy(float) / (l0 + 1) * (1 + uplift)
    z = rec.z.to_numpy(float) if optimised else np.full(len(rec), norm.ppf(service_level))
    sigma = rec.sigma_cover.to_numpy(float) * (1 + uplift) * np.sqrt((lead + 1) / (l0 + 1))
    ss = z * sigma
    up_to = weekly * (lead + 1) + ss
    pos = (rec.on_hand + rec.on_order).to_numpy(float)
    raw = np.clip(up_to - pos, 0, None)
    case, moq = rec.case_pack.to_numpy(float), rec.moq.to_numpy(float)
    qty = np.where(raw > 0, np.maximum(moq, np.ceil(raw / case) * case), 0.0)
    cost = rec.unit_cost.to_numpy(float)
    short = sigma * normal_loss(z)
    status = np.select([qty > 0, pos > 1.3 * up_to], ["Order", "Overstocked"], default="Hold")
    return pd.DataFrame({
        "sku_id": rec.sku_id.to_numpy(), "category": rec.category.to_numpy(), "abc_class": rec.abc_class.to_numpy(),
        "weekly_demand": weekly, "lead_time": lead, "z": z, "safety_stock": ss, "order_up_to": up_to,
        "inventory_position": pos, "order_qty": qty, "order_value": qty * cost, "ss_value": ss * cost,
        "ss_holding_cost": ss * cost * holding_rate, "exp_short_per_week": short, "status": status,
    })


def totals(s: pd.DataFrame) -> dict:
    return {
        "skus_to_order": int((s.status == "Order").sum()),
        "order_value": float(s.order_value.sum()),
        "ss_value": float(s.ss_value.sum()),
        "ss_holding_cost": float(s.ss_holding_cost.sum()),
        "expected_fill": float(1 - s.exp_short_per_week.sum() / s.weekly_demand.sum()),
    }
