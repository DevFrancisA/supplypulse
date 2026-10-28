"""Synthetic retail demand generator.

The generator is deliberately realistic enough to make forecasting non-trivial:
category seasonality, a holiday bump, per-SKU trend, price promotions with a
post-promo dip, and over-dispersed (negative-binomial) noise. The *true* weekly
mean is returned separately so the Monte Carlo simulation can draw fresh demand
paths from the same process without ever feeding it to the forecaster.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .config import SQL_DIR, Config

CATEGORY_PROFILES = {
    #                 seasonal amp, peak week, holiday bump
    "Beverages":     dict(amp=0.35, peak=27, holiday=0.05),
    "Snacks":        dict(amp=0.12, peak=48, holiday=0.35),
    "Household":     dict(amp=0.05, peak=10, holiday=0.00),
    "Personal Care": dict(amp=0.10, peak=3,  holiday=0.10),
    "Pet":           dict(amp=0.04, peak=30, holiday=0.05),
    "Seasonal":      dict(amp=0.70, peak=50, holiday=0.60),
}
LEAD_TIME_BY_SUPPLIER = np.array([1, 1, 2, 2, 3, 3, 4, 4])


def _promo_schedule(rng: np.random.Generator, n_weeks: int, p: float):
    flag = np.zeros(n_weeks, dtype=int)
    disc = np.zeros(n_weeks)
    t = 0
    while t < n_weeks:
        if rng.random() < p:
            length = int(rng.integers(1, 3))
            flag[t:t + length] = 1
            disc[t:t + length] = rng.choice([0.10, 0.15, 0.20, 0.30])
            t += length + 3  # cool-down between promotions
        else:
            t += 1
    return flag, disc


def generate(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (skus, weekly_sales, truth)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_skus
    cats = list(CATEGORY_PROFILES)

    cat_idx = np.arange(n) % len(cats)
    rng.shuffle(cat_idx)
    base = np.clip(rng.lognormal(4.0, 0.9, n), 8, 600)
    growth = rng.normal(0.05, 0.08, n)
    unit_cost = np.clip(rng.lognormal(2.0, 0.8, n), 1.0, 120.0).round(2)
    unit_price = (unit_cost * rng.uniform(1.25, 2.0, n)).round(2)
    supplier_id = rng.integers(0, len(LEAD_TIME_BY_SUPPLIER), n)
    case_pack = rng.choice([6, 12, 24], n)
    moq = case_pack * rng.choice([1, 2, 4], n)
    promo_p = rng.uniform(0.03, 0.08, n)
    elasticity = rng.uniform(1.6, 2.8, n)
    dispersion = rng.uniform(10, 35, n)

    sku_ids = [f"SKU-{i + 1:03d}" for i in range(n)]
    skus = pd.DataFrame({
        "sku_id": sku_ids,
        "category": [cats[c] for c in cat_idx],
        "supplier": [f"Supplier {chr(65 + s)}" for s in supplier_id],
        "unit_cost": unit_cost,
        "unit_price": unit_price,
        "lead_time_weeks": LEAD_TIME_BY_SUPPLIER[supplier_id],
        "case_pack": case_pack,
        "moq": moq,
    })

    weeks = pd.date_range(cfg.start_date, periods=cfg.n_weeks, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy().astype(float)
    t = np.arange(cfg.n_weeks)

    sales_parts, truth_parts = [], []
    for i in range(n):
        prof = CATEGORY_PROFILES[cats[cat_idx[i]]]
        season = (1 + prof["amp"] * np.cos(2 * np.pi * (woy - prof["peak"]) / 52.18)
                  + prof["holiday"] * np.exp(-0.5 * ((woy - 51) / 1.5) ** 2))
        trend = np.exp(growth[i] * t / 52)
        flag, disc = _promo_schedule(rng, cfg.n_weeks, promo_p[i])
        lift = (1 - disc) ** (-elasticity[i])
        after = np.r_[0, flag[:-1]] * (1 - flag)
        dip = 1 - 0.15 * after
        mu = base[i] * trend * season * lift * dip

        k = dispersion[i]
        units = rng.negative_binomial(k, k / (k + mu))
        price = (unit_price[i] * (1 - disc)).round(2)

        sales_parts.append(pd.DataFrame({
            "sku_id": sku_ids[i], "week_idx": t,
            "week_start": weeks.strftime("%Y-%m-%d"),
            "units": units.astype(float), "price": price,
            "promo_flag": flag, "discount_pct": disc,
        }))
        truth_parts.append(pd.DataFrame({
            "sku_id": sku_ids[i], "week_idx": t, "true_mean": mu, "dispersion": k,
        }))

    return skus, pd.concat(sales_parts, ignore_index=True), pd.concat(truth_parts, ignore_index=True)


def to_sqlite(conn: sqlite3.Connection, skus: pd.DataFrame, sales: pd.DataFrame) -> None:
    """Create the schema and load skus + weekly_sales."""
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    sku_cols = ["sku_id", "category", "supplier", "unit_cost", "unit_price",
                "lead_time_weeks", "case_pack", "moq"]
    skus[sku_cols].to_sql("skus", conn, if_exists="append", index=False)
    sales.to_sql("weekly_sales", conn, if_exists="append", index=False)
