"""Monte Carlo inventory simulation (periodic-review, order-up-to policy, lost sales).

Timing per week t:
  1. the order placed at the end of week t-1-L arrives (start of week)
  2. demand d_t is realised; unmet demand is lost
  3. at the end of the week the policy reviews its inventory position and
     orders up to S_t, respecting case-pack and MOQ
An order placed at the end of week t therefore protects weeks t+1 .. t+L+1.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def draw_demand(center: np.ndarray, dispersion: np.ndarray, n_paths: int,
                rng: np.random.Generator) -> np.ndarray:
    """Negative-binomial demand paths, shape (n_skus, n_paths, n_weeks).

    center, dispersion: (n_skus, n_weeks) / (n_skus,) -- mean and NB size."""
    k = dispersion[:, None, None]
    mu = center[:, None, :]
    return rng.negative_binomial(k, k / (k + mu), size=(center.shape[0], n_paths, center.shape[1])).astype(float)


def simulate_sku(demand: np.ndarray, order_up_to: np.ndarray, lead_time: int,
                 case_pack: int, moq: int, init_level: float) -> dict:
    """Simulate one SKU. demand: (P, T); order_up_to: (T,). Returns per-path arrays."""
    P, T = demand.shape
    on_hand = np.full(P, float(order_up_to[0]))
    pipeline = deque(np.full(P, init_level) for _ in range(lead_time + 1))
    oh_end, lost, orders = np.zeros((P, T)), np.zeros((P, T)), np.zeros((P, T))
    pre_order_on_hand = pre_order_on_order = None

    for t in range(T):
        on_hand = on_hand + pipeline.popleft()
        sold = np.minimum(on_hand, demand[:, t])
        lost[:, t] = demand[:, t] - sold
        on_hand = on_hand - sold
        oh_end[:, t] = on_hand

        on_order = sum(pipeline)
        if t == T - 1:
            pre_order_on_hand, pre_order_on_order = on_hand.copy(), on_order.copy()
        raw = np.maximum(order_up_to[t] - (on_hand + on_order), 0.0)
        qty = np.where(raw > 0, np.maximum(moq, np.ceil(raw / case_pack) * case_pack), 0.0)
        pipeline.append(qty)
        orders[:, t] = qty

    return {"on_hand": oh_end, "lost": lost, "orders": orders,
            "final_on_hand": pre_order_on_hand, "final_on_order": pre_order_on_order}


def baseline_levels(ma4_next: np.ndarray, lead_time: int, buffer_weeks: float) -> np.ndarray:
    """Rule of thumb: cover the protection interval at the trailing 4-week average,
    plus a flat buffer of `buffer_weeks` regardless of how volatile the SKU is."""
    return ma4_next * (lead_time + 1 + buffer_weeks)


def supplypulse_levels(fc_next: np.ndarray, lead_time: int, z: float, sigma_cover: float) -> np.ndarray:
    """Forecast-driven: (L+1) x next-week forecast + z x forecast-error sigma over the cover."""
    return fc_next * (lead_time + 1) + z * sigma_cover


def summarise(sim: dict, demand: np.ndarray, unit_cost: float, margin: float,
              holding_rate: float, burn_in: int) -> dict:
    """Per-path KPIs for one SKU (arrays of length P)."""
    sl = slice(burn_in, None)
    dem, lost = demand[:, sl], sim["lost"][:, sl]
    inv = sim["on_hand"][:, sl]
    weeks = dem.shape[1]
    return {
        "demand_units": dem.sum(1),
        "lost_units": lost.sum(1),
        "stockout_weeks": (lost > 0).sum(1),
        "weeks": np.full(dem.shape[0], weeks),
        "avg_inventory_units": inv.mean(1),
        "avg_inventory_value": inv.mean(1) * unit_cost,
        "holding_cost": inv.sum(1) * unit_cost * holding_rate / 52.0,
        "shortage_cost": lost.sum(1) * margin,
    }


def aggregate(per_sku: dict[str, dict]) -> pd.DataFrame:
    """Stack per-SKU, per-path KPIs into portfolio-level per-path KPIs."""
    keys = next(iter(per_sku.values())).keys()
    tot = {k: sum(v[k] for v in per_sku.values()) for k in keys}
    df = pd.DataFrame(tot)
    df["fill_rate"] = 1 - df.lost_units / df.demand_units
    df["stockout_rate"] = df.stockout_weeks / df.weeks  # share of SKU-weeks with any lost sales
    df["total_cost"] = df.holding_cost + df.shortage_cost
    return df
