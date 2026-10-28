"""End-to-end pipeline: data -> SQL features -> forecast -> optimise -> simulate -> export.

    python -m supplypulse.pipeline
"""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pandas as pd

from . import excel_model, powerbi
from .config import CONFIG, DATA_DIR, OUT_DIR, SQL_DIR, Config
from .data import generate, to_sqlite
from .features import build_features
from .forecast import (accuracy_by, feature_importance, fit_model, forecast_future,
                       run_backtest, summarise_accuracy)
from .optimize import cost_service_frontier, normal_loss, optimise_safety_stock
from .simulate import (aggregate, baseline_levels, draw_demand, simulate_sku, summarise,
                       supplypulse_levels)

FRONTIER_TARGETS = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def cover_error_sigma(preds: pd.DataFrame, skus: pd.DataFrame) -> pd.Series:
    """RMS error of the (L+1)-week cumulative forecast, per SKU, on the calibration fold.

    Uses the one-step forecast scaled by (L+1) against the realised (L+1)-week demand --
    exactly how the order-up-to level is built -- so it captures multi-step error that a
    plain one-step sigma x sqrt(L+1) would understate."""
    lead = skus.set_index("sku_id")["lead_time_weeks"]
    cal = preds[preds.fold == "calibration"].sort_values(["sku_id", "week_idx"])
    out = {}
    for sid, g in cal.groupby("sku_id"):
        cover = int(lead[sid]) + 1
        actual_cover = g.actual.rolling(cover).sum().shift(-(cover - 1))
        err = (actual_cover - cover * g.fc_gbm).dropna()
        out[sid] = float(np.sqrt((err ** 2).mean())) if len(err) > 5 else float(g.actual.std() * np.sqrt(cover))
    return pd.Series(out, name="sigma_cover")


def run(cfg: Config = CONFIG, verbose: bool = True) -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 1)

    # 1 ---- data + SQL ------------------------------------------------------
    skus, sales, truth = generate(cfg)
    skus.to_csv(DATA_DIR / "skus.csv", index=False)
    sales.to_csv(DATA_DIR / "weekly_sales.csv", index=False)
    truth.to_csv(DATA_DIR / "true_demand_oracle.csv", index=False)
    db_path = OUT_DIR / "supplypulse.db"
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        to_sqlite(conn, skus, sales)
        abc = pd.read_sql_query((SQL_DIR / "abc_classification.sql").read_text(), conn)
    skus = skus.merge(abc[["sku_id", "abc_class", "revenue"]], on="sku_id")
    _log(f"data + SQL: {len(skus)} SKUs x {cfg.n_weeks} weeks", t0)

    # 2 ---- forecast + backtest --------------------------------------------
    feats = build_features(sales, skus)
    preds, (test_model, test_df) = run_backtest(feats, cfg)
    preds = preds.merge(skus[["sku_id", "abc_class"]], on="sku_id")
    acc = summarise_accuracy(preds)
    imp = feature_importance(test_model, test_df)
    _log(f"forecast WAPE: baseline {acc['wape_ma4']:.1%} -> GBM {acc['wape_gbm']:.1%}", t0)

    final_model = fit_model(feats[feats.week_idx >= cfg.warmup_weeks])
    future = forecast_future(final_model, sales, skus, cfg.horizon_weeks)

    # 3 ---- optimise safety stock ------------------------------------------
    sigma_cover = cover_error_sigma(preds, skus)
    lead = skus.set_index("sku_id")["lead_time_weeks"]
    weekly_mean = future[future.h == 1].set_index("sku_id")["forecast"]
    opt_in = skus.set_index("sku_id")[["category", "abc_class", "unit_cost", "unit_price",
                                       "lead_time_weeks", "case_pack", "moq"]].copy()
    opt_in["sigma_cover"] = sigma_cover
    opt_in["mean_weekly"] = preds[preds.fold == "test"].groupby("sku_id").actual.mean()
    opt_in["holding_rate"] = cfg.holding_rate
    opt_in["margin"] = opt_in.unit_price - opt_in.unit_cost
    opt_in = opt_in.reset_index()
    opt = optimise_safety_stock(opt_in, cfg.target_fill_rate, cfg.min_sku_fill_rate)
    frontier = cost_service_frontier(opt_in, FRONTIER_TARGETS, cfg.min_sku_fill_rate)
    _log(f"optimiser: {opt.status.iloc[0]}, portfolio fill {1 - opt.exp_short_per_week.sum() / opt.mean_weekly.sum():.2%}", t0)
    z_by_sku = opt.set_index("sku_id")["z"]

    # 4 ---- Monte Carlo simulation on the held-out year ---------------------
    T = cfg.test_weeks
    test = preds[preds.fold == "test"]
    sku_ids = list(skus.sku_id)
    fc = test.pivot(index="sku_id", columns="week_idx", values="fc_gbm").loc[sku_ids]
    ma = test.pivot(index="sku_id", columns="week_idx", values="fc_ma4").loc[sku_ids]
    tr = truth[truth.week_idx >= cfg.test_start].pivot(index="sku_id", columns="week_idx", values="true_mean").loc[sku_ids]
    disp = truth.groupby("sku_id").dispersion.first().loc[sku_ids].to_numpy()

    demand = draw_demand(tr.to_numpy(), disp, cfg.n_paths, rng)      # (sku, path, week)
    price = skus.set_index("sku_id")
    # decision at the end of week t uses the forecast for week t+1 (the last decision is moot)
    nxt = lambda a: np.r_[a[1:], a[-1]]  # noqa: E731

    def evaluate(level_fn, detail: bool = False):
        per_sku, weekly, state = {}, [], {}
        for i, sid in enumerate(sku_ids):
            L = int(lead[sid]); cost = price.loc[sid, "unit_cost"]; margin = price.loc[sid, "unit_price"] - cost
            S = level_fn(sid, L, nxt(fc.iloc[i].to_numpy()), nxt(ma.iloc[i].to_numpy()))
            sim = simulate_sku(demand[i], S, L, int(price.loc[sid, "case_pack"]),
                               int(price.loc[sid, "moq"]), float(fc.iloc[i, 0]))
            per_sku[sid] = summarise(sim, demand[i], cost, margin, cfg.holding_rate, cfg.burn_in_weeks)
            if detail:
                state[sid] = (float(sim["final_on_hand"].mean()), float(sim["final_on_order"].mean()))
                weekly.append(pd.DataFrame({
                    "sku_id": sid, "week_idx": np.arange(cfg.test_start, cfg.n_weeks),
                    "avg_on_hand": sim["on_hand"].mean(0), "avg_lost": sim["lost"].mean(0),
                    "avg_demand": demand[i].mean(0), "avg_inventory_value": sim["on_hand"].mean(0) * cost}))
        return per_sku, weekly, state

    def sp_levels(z_series):
        return lambda sid, L, f, m: supplypulse_levels(f, L, z_series[sid], sigma_cover[sid])

    def base_levels(buffer_weeks):
        return lambda sid, L, f, m: baseline_levels(m, L, buffer_weeks)

    per_policy, weekly_rows, sim_state = {}, [], {}
    per_policy["baseline"], w, _ = evaluate(base_levels(cfg.baseline_buffer_weeks), detail=True)
    weekly_rows.append(pd.concat(w).assign(policy="baseline"))
    per_policy["supplypulse"], w, sim_state = evaluate(sp_levels(z_by_sku), detail=True)
    weekly_rows.append(pd.concat(w).assign(policy="supplypulse"))

    paths = {name: aggregate(v) for name, v in per_policy.items()}
    sim_paths = pd.concat([d.assign(policy=n).rename_axis("path").reset_index() for n, d in paths.items()])
    sim_weekly = pd.concat(weekly_rows, ignore_index=True)
    sim_by_sku = pd.DataFrame([
        {"sku_id": sid, "policy": name,
         "fill_rate": 1 - v["lost_units"].sum() / v["demand_units"].sum(),
         "stockout_rate": v["stockout_weeks"].sum() / v["weeks"].sum(),
         "avg_inventory_value": float(v["avg_inventory_value"].mean()),
         "total_cost": float((v["holding_cost"] + v["shortage_cost"]).mean())}
        for name, d in per_policy.items() for sid, v in d.items()
    ])

    # cost / service frontier: sweep the rule-of-thumb buffer and the optimiser's fill target
    def point(policy, param, per_sku, extra=None):
        a = aggregate(per_sku)
        return {"policy": policy, "parameter": param, "fill_rate": a.fill_rate.mean(),
                "stockout_rate": a.stockout_rate.mean(), "avg_inventory_value": a.avg_inventory_value.mean(),
                "total_cost": a.total_cost.mean(), **(extra or {})}

    frontier_rows = []
    for buf in np.arange(0.0, 6.01, 0.5):
        frontier_rows.append(point("Rule of thumb", f"{buf:g} wk buffer", evaluate(base_levels(buf))[0]))
    for tgt in FRONTIER_TARGETS:
        o = optimise_safety_stock(opt_in, tgt, cfg.min_sku_fill_rate, penalty_weight=0.0)
        frontier_rows.append(point("SupplyPulse", f"{tgt:.1%} target", evaluate(sp_levels(o.set_index("sku_id")["z"]))[0]))
    policy_frontier = pd.DataFrame(frontier_rows)
    _log(f"simulation: {cfg.n_paths} paths x {len(sku_ids)} SKUs x {T} weeks, {2 + len(frontier_rows)} policy runs", t0)

    b, s = paths["baseline"], paths["supplypulse"]
    kpi = {
        "baseline": {m: float(b[m].mean()) for m in ["fill_rate", "stockout_rate", "lost_units", "avg_inventory_value", "holding_cost", "shortage_cost", "total_cost"]},
        "supplypulse": {m: float(s[m].mean()) for m in ["fill_rate", "stockout_rate", "lost_units", "avg_inventory_value", "holding_cost", "shortage_cost", "total_cost"]},
    }
    red = lambda m: 100 * (1 - kpi["supplypulse"][m] / kpi["baseline"][m])  # noqa: E731
    # paired comparison (same demand path index) for uncertainty bands
    paired = pd.DataFrame({
        "stockout_reduction_pct": 100 * (1 - s.stockout_weeks.to_numpy() / b.stockout_weeks.to_numpy()),
        "lost_units_reduction_pct": 100 * (1 - s.lost_units.to_numpy() / b.lost_units.to_numpy()),
        "inventory_reduction_pct": 100 * (1 - s.avg_inventory_value.to_numpy() / b.avg_inventory_value.to_numpy()),
    })
    # inventory needed by the rule of thumb to match SupplyPulse's simulated fill rate (interpolated)
    rot = policy_frontier[policy_frontier.policy == "Rule of thumb"].sort_values("fill_rate")
    sp_fill = kpi["supplypulse"]["fill_rate"]
    if rot.fill_rate.min() <= sp_fill <= rot.fill_rate.max():
        matched_inv = float(np.interp(sp_fill, rot.fill_rate, rot.avg_inventory_value))
        matched_reduction = 100 * (1 - kpi["supplypulse"]["avg_inventory_value"] / matched_inv)
    else:
        matched_reduction = None

    # 5 ---- replenishment recommendations ----------------------------------
    rec = opt[["sku_id", "category", "abc_class", "unit_cost", "unit_price", "lead_time_weeks",
               "case_pack", "moq", "sigma_cover", "z", "safety_stock", "fill_rate"]].copy()
    rec = rec.rename(columns={"fill_rate": "expected_fill_rate"})
    rec["on_hand"] = rec.sku_id.map(lambda k: sim_state[k][0])
    rec["on_order"] = rec.sku_id.map(lambda k: sim_state[k][1])
    cover_fc = []
    for r in rec.itertuples():
        f = future[future.sku_id == r.sku_id].sort_values("h").forecast.to_numpy()
        cover_fc.append(f[: int(r.lead_time_weeks) + 1].sum())
    rec["weekly_forecast"] = rec.sku_id.map(weekly_mean)
    rec["cover_demand"] = cover_fc
    rec["order_up_to"] = rec.cover_demand + rec.safety_stock
    rec["inventory_position"] = rec.on_hand + rec.on_order
    raw = (rec.order_up_to - rec.inventory_position).clip(lower=0)
    rec["order_qty"] = np.where(raw > 0, np.maximum(rec.moq, np.ceil(raw / rec.case_pack) * rec.case_pack), 0).astype(int)
    rec["order_value"] = rec.order_qty * rec.unit_cost
    rec["weeks_of_cover"] = rec.on_hand / rec.weekly_forecast.clip(lower=1e-9)
    rec["status"] = np.select(
        [rec.order_qty > 0, rec.inventory_position > 1.3 * rec.order_up_to],
        ["Order", "Overstocked"], default="Hold")
    rec = rec.round({"sigma_cover": 1, "safety_stock": 1, "on_hand": 1, "on_order": 1, "weekly_forecast": 1,
                     "cover_demand": 1, "order_up_to": 1, "inventory_position": 1, "weeks_of_cover": 2,
                     "expected_fill_rate": 4, "order_value": 2})

    # 6 ---- persist ---------------------------------------------------------
    metrics = {
        "generated_for": {"n_skus": cfg.n_skus, "n_weeks": cfg.n_weeks, "test_weeks": cfg.test_weeks,
                          "n_paths": cfg.n_paths, "seed": cfg.seed},
        "forecast": acc,
        "simulation": {
            "baseline": kpi["baseline"], "supplypulse": kpi["supplypulse"],
            "stockout_reduction_pct": red("stockout_rate"),
            "lost_units_reduction_pct": red("lost_units"),
            "inventory_reduction_pct": red("avg_inventory_value"),
            "total_cost_reduction_pct": red("total_cost"),
            "inventory_reduction_at_matched_fill_pct": matched_reduction,
            "paired_p05_p95": {c: [float(paired[c].quantile(.05)), float(paired[c].quantile(.95))] for c in paired},
        },
        "optimiser": {"status": opt.status.iloc[0], "target_fill_rate": cfg.target_fill_rate,
                      "expected_fill_rate": float(1 - opt.exp_short_per_week.sum() / opt.mean_weekly.sum()),
                      "safety_stock_value": float((opt.safety_stock * opt.unit_cost).sum())},
        "baseline_policy": f"order-up-to = 4-wk moving average x (L+1+{cfg.baseline_buffer_weeks:g}) weeks",
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_csv(OUT_DIR / "forecasts.csv", index=False)
    future.to_csv(OUT_DIR / "future_forecast.csv", index=False)
    rec.to_csv(OUT_DIR / "recommendations.csv", index=False)
    opt.to_csv(OUT_DIR / "optimizer.csv", index=False)
    frontier.to_csv(OUT_DIR / "frontier.csv", index=False)
    policy_frontier.to_csv(OUT_DIR / "policy_frontier.csv", index=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    accuracy_by(preds, "category").to_csv(OUT_DIR / "accuracy_by_category.csv", index=False)
    accuracy_by(preds, "sku_id").to_csv(OUT_DIR / "accuracy_by_sku.csv", index=False)
    sim_paths.to_csv(OUT_DIR / "sim_paths.csv", index=False)
    sim_weekly.to_csv(OUT_DIR / "sim_weekly.csv", index=False)
    sim_by_sku.to_csv(OUT_DIR / "sim_by_sku.csv", index=False)
    skus.to_csv(OUT_DIR / "skus.csv", index=False)
    sales.to_csv(OUT_DIR / "weekly_sales.csv", index=False)
    with sqlite3.connect(db_path) as conn:
        for name, df in [("forecasts", preds), ("recommendations", rec), ("optimizer", opt),
                         ("sim_by_sku", sim_by_sku)]:
            df.to_sql(name, conn, if_exists="replace", index=False)
    _log("saved outputs/", t0)

    excel_model.build(rec, OUT_DIR / "SupplyPulse_WhatIf_Model.xlsx", cfg)
    powerbi.export(skus, sales, preds, future, rec, sim_by_sku, sim_weekly, sim_paths, metrics, cfg)
    _log("excel what-if model + Power BI exports written", t0)

    if verbose:
        sm = metrics["simulation"]
        print("\n=== SupplyPulse results ===")
        print(f"WAPE (held-out year): baseline {acc['wape_ma4']:.1%} -> model {acc['wape_gbm']:.1%} "
              f"({acc['wape_improvement_pct']:.1f}% lower)")
        print(f"Stockout SKU-weeks:   {sm['stockout_reduction_pct']:+.1f}% fewer | lost units {sm['lost_units_reduction_pct']:+.1f}% fewer")
        print(f"Average inventory:    {sm['inventory_reduction_pct']:+.1f}% lower | total cost {sm['total_cost_reduction_pct']:+.1f}% lower")
        print(f"Fill rate:            {kpi['baseline']['fill_rate']:.2%} -> {kpi['supplypulse']['fill_rate']:.2%}")
        if matched_reduction is not None:
            print(f"Inventory at matched fill rate: {matched_reduction:+.1f}% lower than the rule of thumb needs")
    return metrics


if __name__ == "__main__":
    run()
