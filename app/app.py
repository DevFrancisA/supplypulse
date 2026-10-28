"""SupplyPulse - local planner dashboard.   Run:  streamlit run app/app.py"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supplypulse.config import CONFIG, OUT_DIR, SQL_DIR  # noqa: E402
from supplypulse.optimize import normal_loss, optimise_safety_stock  # noqa: E402
from supplypulse.whatif import scenario, totals  # noqa: E402

st.set_page_config(page_title="SupplyPulse", page_icon="📦", layout="wide")

TEAL, SLATE, AMBER, INK, MUTED, RED = "#0F766E", "#94A3B8", "#D97706", "#1F2937", "#6B7280", "#B42318"
POLICY_COLOR = {"baseline": SLATE, "supplypulse": TEAL, "Rule of thumb": SLATE, "SupplyPulse": TEAL}
POLICY_LABEL = {"baseline": "Rule of thumb", "supplypulse": "SupplyPulse"}

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1280px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .kpi {background:#fff; border:1px solid transparent; border-radius:14px; padding:18px 20px; min-height:158px;
        box-shadow: 0 0 0 1px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.04);}
  .kpi .label {font-size:.78rem; color:#6B7280; text-transform:uppercase; letter-spacing:.06em; font-weight:600;}
  .kpi .value {font-size:2rem; font-weight:700; color:#1F2937; font-variant-numeric: tabular-nums; line-height:1.15; margin-top:6px;}
  .kpi .value.good {color:#0F766E;}
  .kpi .sub {font-size:.84rem; color:#6B7280; margin-top:4px; font-variant-numeric: tabular-nums;}
  .stTabs [data-baseweb="tab"] {font-weight:600;}
  div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
  @media (prefers-reduced-motion: no-preference) {
    .kpi {transition: box-shadow 150ms ease-out;}
  }
  @media (hover: hover) {
    .kpi:hover {box-shadow: 0 0 0 1px rgba(15,23,42,.10), 0 2px 4px rgba(15,23,42,.08), 0 8px 20px rgba(15,23,42,.07);}
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="First run: building the data, models and simulation (about 10 seconds)...")
def ensure_outputs() -> bool:
    if not (OUT_DIR / "metrics.json").exists() or not (OUT_DIR / "supplypulse.db").exists():
        from supplypulse.pipeline import run
        run(CONFIG, verbose=False)
    return True


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / name)


@st.cache_data
def load_metrics() -> dict:
    return json.loads((OUT_DIR / "metrics.json").read_text())


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=13),
                      legend=dict(orientation="h", y=1.12, x=0, title=None),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1", zeroline=False)
    fig.update_yaxes(gridcolor="#E5E7EB", zeroline=False)
    return fig


def kpi(col, label, value, sub="", good=False):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value {"good" if good else ""}">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


ensure_outputs()
M = load_metrics()
fc, rec, skus = load("forecasts.csv"), load("recommendations.csv"), load("skus.csv")
sales = load("weekly_sales.csv")
cfg_meta = M["generated_for"]

st.title("SupplyPulse")
st.caption(f"Demand forecasting and replenishment planning - {cfg_meta['n_skus']} SKUs, {cfg_meta['n_weeks']} weeks of "
           f"synthetic weekly history, {cfg_meta['n_paths']} Monte Carlo paths. Python - SQL - scikit-learn - PuLP.")

tab_over, tab_fc, tab_rep, tab_sim, tab_opt, tab_sql = st.tabs(
    ["Overview", "Forecast", "Replenishment", "Simulation", "Optimizer", "Data & SQL"])

# ------------------------------------------------------------------ overview
with tab_over:
    f, s = M["forecast"], M["simulation"]
    c = st.columns(4)
    kpi(c[0], "Forecast error (WAPE)", f"{f['wape_gbm']:.1%}",
        f"vs {f['wape_ma4']:.1%} moving average ({f['wape_improvement_pct']:.0f}% lower)", good=True)
    kpi(c[1], "Stockout events", f"-{s['stockout_reduction_pct']:.0f}%",
        f"SKU-weeks with lost sales: {s['baseline']['stockout_rate']:.1%} to {s['supplypulse']['stockout_rate']:.1%}", good=True)
    mr = s.get("inventory_reduction_at_matched_fill_pct")
    kpi(c[2], "Inventory at equal service", f"-{mr:.0f}%" if mr is not None else "n/a",
        f"fill rate held at {s['supplypulse']['fill_rate']:.1%}", good=True)
    kpi(c[3], "Holding + lost-sales cost", f"-{s['total_cost_reduction_pct']:.0f}%",
        f"${s['baseline']['total_cost']:,.0f} to ${s['supplypulse']['total_cost']:,.0f} over the test year", good=True)

    st.markdown("&nbsp;")
    left, right = st.columns([3, 2], gap="large")
    pf = load("policy_frontier.csv")
    with left:
        st.subheader("Cost / service frontier")
        fig = go.Figure()
        for pol, g in pf.groupby("policy"):
            g = g.sort_values("avg_inventory_value")
            fig.add_scatter(x=g.avg_inventory_value, y=g.fill_rate, mode="lines+markers", name=pol,
                            line=dict(color=POLICY_COLOR[pol], width=3), marker=dict(size=7),
                            customdata=g.parameter, hovertemplate="%{customdata}<br>inventory $%{x:,.0f}<br>fill %{y:.2%}<extra></extra>")
        b, sp = s["baseline"], s["supplypulse"]
        fig.add_scatter(x=[b["avg_inventory_value"]], y=[b["fill_rate"]], mode="markers+text", text=["current rule: 2-week buffer"],
                        textposition="bottom right", marker=dict(size=14, color="white", line=dict(color=SLATE, width=3)),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[sp["avg_inventory_value"]], y=[sp["fill_rate"]], mode="markers+text", text=["SupplyPulse (cost-optimal)"],
                        textposition="top right", marker=dict(size=14, color="white", line=dict(color=TEAL, width=3)),
                        showlegend=False, hoverinfo="skip")
        fig.update_yaxes(tickformat=".0%", title="Simulated fill rate", range=[0.88, 1.002])
        fig.update_xaxes(title="Average inventory value ($)", tickprefix="$", tickformat=",", range=[40000, 215000])
        st.plotly_chart(style(fig, 420), width="stretch")
        st.caption("Up and to the left is better: more service for less inventory. Each point is a full 300-path simulation. "
                   "The rule of thumb is swept from 0 to 6 weeks of flat safety cover; SupplyPulse is swept across portfolio fill targets.")
    with right:
        st.subheader("What drives the forecast")
        imp = load("feature_importance.csv").head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.importance, y=imp.feature, orientation="h", marker_color=TEAL))
        fig.update_xaxes(title="Permutation importance (MAE increase)")
        st.plotly_chart(style(fig, 420), width="stretch")

    with st.expander("How to read these numbers (and what they do not prove)"):
        st.markdown(f"""
- **Data is synthetic.** It has seasonality, trend, promotions with lift and post-promo dips, and over-dispersed noise, so the
  forecasting problem is real, but absolute numbers are not a claim about any real business.
- **WAPE** = sum of absolute errors / sum of actual demand, measured on a held-out final year the model never trained on.
  Baseline = trailing 4-week moving average. Promo-week WAPE: {f['wape_gbm_promo_weeks']:.1%} (model) vs {f['wape_ma4_promo_weeks']:.1%} (baseline).
- **Simulation** replays the held-out year over {cfg_meta['n_paths']} demand paths drawn from the data-generating process, identical for both policies.
  Rule of thumb = {M['baseline_policy']}. Stockouts are lost sales.
- **Equal-service inventory** compares against the rule-of-thumb buffer that would deliver the same simulated fill rate (interpolated on the swept curve),
  which is a fairer comparison than a single arbitrary buffer. At the fixed 2-week buffer alone, average inventory is {s['inventory_reduction_pct']:.1f}% lower
  while stockout events fall {s['stockout_reduction_pct']:.0f}%.
- The optimiser's analytic fill rate ({M['optimiser']['expected_fill_rate']:.2%}) is more optimistic than the simulated one ({s['supplypulse']['fill_rate']:.2%}):
  a normal-error model understates skew, MOQ effects and multi-step forecast error. The simulation is the arbiter.
- Paired 5-95% range across paths for stockout reduction: {s['paired_p05_p95']['stockout_reduction_pct'][0]:.0f}% to {s['paired_p05_p95']['stockout_reduction_pct'][1]:.0f}%.
""")

# ------------------------------------------------------------------ forecast
with tab_fc:
    fut = load("future_forecast.csv")
    a, b_, c_ = st.columns([2, 2, 3])
    cat = a.selectbox("Category", ["All"] + sorted(skus.category.unique()))
    pool = skus if cat == "All" else skus[skus.category == cat]
    sku = b_.selectbox("SKU", list(pool.sku_id))
    scope = c_.radio("Show", ["Held-out year", "Calibration + held-out"], horizontal=True, index=1)
    g = fc[fc.sku_id == sku].copy()
    if scope == "Held-out year":
        g = g[g.fold == "test"]
    g["date"] = pd.to_datetime(g.week_start)
    gf = fut[fut.sku_id == sku].copy()
    gf["date"] = pd.to_datetime(gf.week_start)

    t = g[g.fold == "test"]
    w_g = np.abs(t.actual - t.fc_gbm).sum() / t.actual.sum()
    w_m = np.abs(t.actual - t.fc_ma4).sum() / t.actual.sum()
    info = skus[skus.sku_id == sku].iloc[0]
    m = st.columns(5)
    m[0].metric("WAPE - model", f"{w_g:.1%}")
    m[1].metric("WAPE - 4-wk avg", f"{w_m:.1%}", f"{(w_g - w_m) * 100:+.1f} pts", delta_color="inverse")
    m[2].metric("Lead time", f"{info.lead_time_weeks} wk")
    m[3].metric("ABC class", info.abc_class)
    m[4].metric("Unit price", f"${info.unit_price:,.2f}")

    fig = go.Figure()
    fig.add_scatter(x=g.date, y=g.actual, name="Actual", mode="lines", line=dict(color=INK, width=2))
    fig.add_scatter(x=g.date, y=g.fc_ma4, name="4-wk moving average", mode="lines", line=dict(color=SLATE, width=2, dash="dash"))
    fig.add_scatter(x=g.date, y=g.fc_gbm, name="Gradient boosting", mode="lines", line=dict(color=TEAL, width=3))
    pr = g[g.promo_flag == 1]
    fig.add_scatter(x=pr.date, y=pr.actual, name="Promo week", mode="markers", marker=dict(color=AMBER, size=10, symbol="diamond"))
    fig.add_scatter(x=gf.date, y=gf.forecast, name="Next 6 weeks", mode="lines+markers",
                    line=dict(color=TEAL, width=3, dash="dot"), marker=dict(size=6))
    test_start = pd.to_datetime(fc[(fc.fold == "test")].week_start.min())
    fig.add_vrect(x0=test_start, x1=g.date.max(), fillcolor=TEAL, opacity=0.05, line_width=0)
    fig.update_yaxes(title="Units per week")
    st.plotly_chart(style(fig, 440), width="stretch")
    st.caption("Shaded region: held-out year. The dotted line is a recursive forecast beyond the data (no planned promotions assumed).")

    l, r = st.columns(2, gap="large")
    with l:
        st.subheader("Accuracy by category")
        ac = load("accuracy_by_category.csv").sort_values("volume", ascending=False)
        fig = go.Figure()
        fig.add_bar(x=ac.category, y=ac.wape_ma4, name="4-wk moving average", marker_color=SLATE)
        fig.add_bar(x=ac.category, y=ac.wape_gbm, name="Gradient boosting", marker_color=TEAL)
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", title="WAPE")
        st.plotly_chart(style(fig, 340), width="stretch")
    with r:
        st.subheader("Where the model helps most")
        sk = load("accuracy_by_sku.csv")
        fig = go.Figure(go.Histogram(x=sk.improvement_pct, nbinsx=18, marker_color=TEAL))
        fig.add_vline(x=0, line_color=INK, line_width=1)
        fig.update_xaxes(title="WAPE improvement vs. baseline (%)")
        fig.update_yaxes(title="SKUs")
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption(f"{(sk.improvement_pct > 0).sum()} of {len(sk)} SKUs are more accurate than the moving average.")

# ------------------------------------------------------------------ replenishment
with tab_rep:
    st.subheader("This week's replenishment plan")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        mode = c1.radio("Service level", ["Optimised per SKU", "One level for all"], help="Optimised = PuLP solution (holding vs. lost-sales cost).")
        sl = c1.slider("Global service level", 0.80, 0.999, 0.95, 0.005, disabled=mode == "Optimised per SKU", format="%.3f")
        uplift = c2.slider("Demand vs. forecast", -30, 50, 0, 1, format="%d%%") / 100
        delay = c3.slider("Supplier delay (weeks)", 0, 6, 0)
        hold = c4.slider("Annual holding rate", 0.10, 0.40, float(CONFIG.holding_rate), 0.01, format="%.2f")

    base = totals(scenario(rec, True, 0.95, 0.0, 0, hold))
    sc = scenario(rec, mode == "Optimised per SKU", sl, uplift, delay, hold)
    tt = totals(sc)
    def dl(diff, fmt):
        """Delta label, or None when the scenario equals the base (avoids a misleading arrow)."""
        return None if abs(diff) < 1e-9 else format(diff, fmt)

    k = st.columns(5)
    k[0].metric("SKUs to order", tt["skus_to_order"], dl(tt["skus_to_order"] - base["skus_to_order"], "+d") and f"{tt['skus_to_order'] - base['skus_to_order']:+d} vs base")
    k[1].metric("Order value", f"${tt['order_value']:,.0f}", dl(tt["order_value"] - base["order_value"], "+,.0f"))
    k[2].metric("Safety stock value", f"${tt['ss_value']:,.0f}", dl(tt["ss_value"] - base["ss_value"], "+,.0f"), delta_color="inverse")
    k[3].metric("Holding cost of safety stock / yr", f"${tt['ss_holding_cost']:,.0f}", dl(tt["ss_holding_cost"] - base["ss_holding_cost"], "+,.0f"), delta_color="inverse")
    k[4].metric("Expected fill rate", f"{tt['expected_fill']:.2%}", dl((tt["expected_fill"] - base["expected_fill"]) * 100, "+.2f") and f"{(tt['expected_fill'] - base['expected_fill']) * 100:+.2f} pts")

    f1, f2, f3 = st.columns(3)
    stat = f1.multiselect("Status", ["Order", "Hold", "Overstocked"], default=["Order", "Hold", "Overstocked"])
    cats = f2.multiselect("Category", sorted(skus.category.unique()))
    abc = f3.multiselect("ABC class", ["A", "B", "C"])
    view = sc[sc.status.isin(stat)]
    if cats:
        view = view[view.category.isin(cats)]
    if abc:
        view = view[view.abc_class.isin(abc)]
    view = view.merge(rec[["sku_id", "on_hand", "on_order", "case_pack", "moq"]], on="sku_id")
    show = view[["sku_id", "category", "abc_class", "status", "order_qty", "order_value", "order_up_to", "inventory_position",
                 "on_hand", "on_order", "safety_stock", "z", "lead_time", "case_pack", "moq"]].sort_values("order_value", ascending=False)
    st.dataframe(show, width="stretch", hide_index=True, height=420, column_config={
        "sku_id": "SKU", "category": "Category", "abc_class": "ABC", "status": "Status",
        "order_qty": st.column_config.NumberColumn("Order qty", format="%d"),
        "order_value": st.column_config.NumberColumn("Order value", format="$%,.0f"),
        "order_up_to": st.column_config.NumberColumn("Order-up-to", format="%.0f"),
        "inventory_position": st.column_config.NumberColumn("Inv. position", format="%.0f"),
        "on_hand": st.column_config.NumberColumn("On hand", format="%.0f"),
        "on_order": st.column_config.NumberColumn("On order", format="%.0f"),
        "safety_stock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
        "z": st.column_config.NumberColumn("z", format="%.2f"),
        "lead_time": st.column_config.NumberColumn("Lead (wk)", format="%d"),
        "case_pack": "Case", "moq": "MOQ"})
    d1, d2 = st.columns([1, 4])
    d1.download_button("Download plan (CSV)", show.to_csv(index=False).encode(), "supplypulse_plan.csv", "text/csv")
    xlsx = OUT_DIR / "SupplyPulse_WhatIf_Model.xlsx"
    if xlsx.exists():
        d2.download_button("Download Excel what-if model", xlsx.read_bytes(), xlsx.name,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("&nbsp;")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.subheader("Order value by category")
        by = sc.groupby("category", as_index=False).order_value.sum().sort_values("order_value")
        fig = go.Figure(go.Bar(x=by.order_value, y=by.category, orientation="h", marker_color=TEAL))
        fig.update_xaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(style(fig, 320), width="stretch")
    with g2:
        st.subheader("Safety stock vs. variability")
        d = sc.merge(rec[["sku_id", "sigma_cover"]], on="sku_id")
        fig = go.Figure()
        for k_, grp in d.groupby("abc_class"):
            fig.add_scatter(x=grp.sigma_cover, y=grp.ss_value, mode="markers", name=f"Class {k_}", text=grp.sku_id,
                            marker=dict(size=10, color={"A": TEAL, "B": AMBER, "C": SLATE}[k_]))
        fig.update_xaxes(title="Forecast-error sigma over cover (units)")
        fig.update_yaxes(title="Safety stock value ($)", tickprefix="$", tickformat=",")
        st.plotly_chart(style(fig, 320), width="stretch")

# ------------------------------------------------------------------ simulation
with tab_sim:
    paths, sw, bysku = load("sim_paths.csv"), load("sim_weekly.csv"), load("sim_by_sku.csv")
    st.subheader("Policy comparison across Monte Carlo paths")
    metric_opts = {"Stockout rate (share of SKU-weeks)": ("stockout_rate", ".1%"), "Fill rate": ("fill_rate", ".2%"),
                   "Average inventory value ($)": ("avg_inventory_value", ",.0f"), "Total cost ($)": ("total_cost", ",.0f")}
    cols = st.columns(len(metric_opts))
    for col, (label, (key, fmt)) in zip(cols, metric_opts.items()):
        fig = go.Figure()
        for pol in ["baseline", "supplypulse"]:
            v = paths[paths.policy == pol][key]
            fig.add_box(y=v, name=POLICY_LABEL[pol], marker_color=POLICY_COLOR[pol], boxpoints=False, line_width=2)
        fig.update_layout(showlegend=False, title=dict(text=label, font=dict(size=13)))
        fig.update_yaxes(tickformat=fmt)
        col.plotly_chart(style(fig, 330), width="stretch")
    st.caption("Each box summarises the portfolio outcome across demand paths; both policies face identical paths.")

    st.subheader("Inventory and lost sales over the held-out year")
    pick = st.selectbox("SKU", ["All SKUs"] + list(skus.sku_id), key="sim_sku")
    d = sw if pick == "All SKUs" else sw[sw.sku_id == pick]
    col_inv = "avg_inventory_value" if pick == "All SKUs" else "avg_on_hand"
    wk = d.groupby(["policy", "week_idx"], as_index=False).agg(inv=(col_inv, "sum"), lost=("avg_lost", "sum"), demand=("avg_demand", "sum"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                        subplot_titles=("Average inventory " + ("($)" if pick == "All SKUs" else "(units)"), "Average lost sales (units)"))
    for pol in ["baseline", "supplypulse"]:
        g = wk[wk.policy == pol]
        fig.add_scatter(x=g.week_idx, y=g.inv, name=POLICY_LABEL[pol], line=dict(color=POLICY_COLOR[pol], width=3), row=1, col=1)
        fig.add_bar(x=g.week_idx, y=g.lost, name=POLICY_LABEL[pol], marker_color=POLICY_COLOR[pol], showlegend=False, row=2, col=1)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Week index", row=2, col=1)
    st.plotly_chart(style(fig, 520), width="stretch")

    st.subheader("Per-SKU outcome")
    pv = bysku.pivot(index="sku_id", columns="policy")
    tbl = pd.DataFrame({
        "Fill rate - rule": pv[("fill_rate", "baseline")], "Fill rate - SupplyPulse": pv[("fill_rate", "supplypulse")],
        "Inventory $ - rule": pv[("avg_inventory_value", "baseline")], "Inventory $ - SupplyPulse": pv[("avg_inventory_value", "supplypulse")],
        "Cost $ - rule": pv[("total_cost", "baseline")], "Cost $ - SupplyPulse": pv[("total_cost", "supplypulse")]}).reset_index()
    tbl["Cost saved $"] = tbl["Cost $ - rule"] - tbl["Cost $ - SupplyPulse"]
    st.dataframe(tbl.sort_values("Cost saved $", ascending=False), width="stretch", hide_index=True, height=360,
                 column_config={c: st.column_config.NumberColumn(format="%.3f" if "Fill" in c else "$%,.0f")
                                for c in tbl.columns if c != "sku_id"})

# ------------------------------------------------------------------ optimizer
with tab_opt:
    st.subheader("Safety-stock optimiser (PuLP, solved live)")
    st.markdown("Each SKU picks one service level from a grid of 31 options. The solver minimises cost subject to fill-rate floors.")
    o = load("optimizer.csv")
    inputs = o[["sku_id", "category", "abc_class", "mean_weekly", "sigma_cover", "unit_cost", "holding_rate", "margin"]]
    c1, c2, c3 = st.columns(3)
    obj = c1.radio("Objective", ["Holding + lost-sales cost", "Holding cost only (service-constrained)"])
    tgt = c2.slider("Portfolio fill-rate target", 0.90, 0.995, float(CONFIG.target_fill_rate), 0.005, format="%.3f")
    floor = c3.slider("Per-SKU fill-rate floor", 0.70, 0.97, float(CONFIG.min_sku_fill_rate), 0.01, format="%.2f")

    @st.cache_data(show_spinner="Solving...")
    def solve(obj_flag: float, tgt: float, floor: float) -> pd.DataFrame:
        return optimise_safety_stock(inputs, tgt, floor, penalty_weight=obj_flag)

    r = solve(1.0 if obj.startswith("Holding +") else 0.0, tgt, floor)
    fill = 1 - r.exp_short_per_week.sum() / r.mean_weekly.sum()
    ssv = float((r.safety_stock * r.unit_cost).sum())
    # same expected fill achieved with a single z for all SKUs
    zs = np.linspace(0, 3.5, 351)
    uni_fill = np.array([1 - (r.sigma_cover * normal_loss(z)).sum() / r.mean_weekly.sum() for z in zs])
    z_u = zs[np.argmax(uni_fill >= fill)]
    uni_val = float((r.sigma_cover * z_u * r.unit_cost).sum())
    k = st.columns(4)
    k[0].metric("Solver status", r.status.iloc[0])
    k[1].metric("Expected fill rate", f"{fill:.2%}")
    k[2].metric("Safety stock value", f"${ssv:,.0f}")
    k[3].metric("vs. one z for all SKUs", f"{(1 - ssv / uni_val):.0%} less", f"same fill rate (z = {z_u:.2f})", delta_color="off")

    a, b_ = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        for k_, grp in r.groupby("abc_class"):
            fig.add_scatter(x=grp.sigma_cover / grp.mean_weekly, y=grp.z, mode="markers", name=f"Class {k_}", text=grp.sku_id,
                            marker=dict(size=np.sqrt(grp.unit_cost) * 3, color={"A": TEAL, "B": AMBER, "C": SLATE}[k_], opacity=.85))
        fig.update_xaxes(title="Forecast uncertainty (sigma / weekly demand)")
        fig.update_yaxes(title="Chosen z (service level)")
        st.plotly_chart(style(fig.update_layout(title=dict(text="Who gets more protection", font=dict(size=13))), 360), width="stretch")
    with b_:
        fr = load("frontier.csv")
        fig = go.Figure(go.Scatter(x=fr.achieved_fill_rate, y=fr.safety_stock_value, mode="lines+markers",
                                   line=dict(color=TEAL, width=3), marker=dict(size=8)))
        fig.add_scatter(x=[fill], y=[ssv], mode="markers", marker=dict(size=14, color="white", line=dict(color=AMBER, width=3)), name="Current", showlegend=False)
        fig.update_xaxes(title="Expected fill rate", tickformat=".1%")
        fig.update_yaxes(title="Safety stock value ($)", tickprefix="$", tickformat=",")
        st.plotly_chart(style(fig.update_layout(title=dict(text="Price of service (holding-cost-only frontier)", font=dict(size=13))), 360), width="stretch")
    st.dataframe(r[["sku_id", "category", "abc_class", "mean_weekly", "sigma_cover", "z", "safety_stock", "fill_rate", "annual_cost"]]
                 .sort_values("annual_cost", ascending=False), width="stretch", hide_index=True, height=300)

# ------------------------------------------------------------------ data & sql
with tab_sql:
    st.subheader("SQL layer")
    st.markdown("Weekly sales live in SQLite (`outputs/supplypulse.db`). Training features are produced by a window-function query; "
                "every lag and rolling window ends at `1 PRECEDING`, so a row never sees its own week or the future.")
    q1, q2 = st.columns(2, gap="large")
    with q1:
        st.markdown("**features.sql**")
        st.code((SQL_DIR / "features.sql").read_text(), language="sql")
    with q2:
        st.markdown("**abc_classification.sql**")
        st.code((SQL_DIR / "abc_classification.sql").read_text(), language="sql")

    st.subheader("Run your own query")
    db = OUT_DIR / "supplypulse.db"
    default = ("SELECT s.category, COUNT(*) AS skus, ROUND(AVG(r.order_qty), 1) AS avg_order_qty,\n"
               "       ROUND(SUM(r.order_value)) AS order_value\n"
               "FROM recommendations r JOIN skus s USING (sku_id)\nGROUP BY s.category ORDER BY order_value DESC;")
    st.caption("Tables: skus, weekly_sales, forecasts, recommendations, optimizer, sim_by_sku. Read-only; SELECT / WITH only.")
    q = st.text_area("SQL", default, height=140, label_visibility="collapsed")
    if st.button("Run query", type="primary"):
        stmt = q.strip().rstrip(";")
        if not re.match(r"(?is)^\s*(select|with)\b", stmt) or ";" in stmt:
            st.error("Only a single SELECT or WITH statement is allowed.")
        else:
            try:
                with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as conn:
                    res = pd.read_sql_query(stmt + " LIMIT 1000" if " limit " not in stmt.lower() else stmt, conn)
                st.dataframe(res, width="stretch", hide_index=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
