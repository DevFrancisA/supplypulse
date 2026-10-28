import numpy as np
import pandas as pd
import pytest

from supplypulse.optimize import Z_GRID, normal_loss, optimise_safety_stock
from supplypulse.simulate import baseline_levels, draw_demand, simulate_sku, supplypulse_levels


def _demand(mu=50.0, T=40, P=200, seed=0):
    rng = np.random.default_rng(seed)
    return draw_demand(np.full((1, T), mu), np.array([20.0]), P, rng)[0]


def test_demand_draws_match_the_mean():
    d = _demand(mu=80, P=2000)
    assert abs(d.mean() - 80) < 2


def test_stock_flow_identity_and_non_negative_inventory():
    """opening stock + arrivals - sales == closing stock, exactly, on every path."""
    T, L, init_level, S0 = 40, 2, 50.0, 200.0
    d = _demand(T=T)
    sim = simulate_sku(d, np.full(T, S0), lead_time=L, case_pack=12, moq=24, init_level=init_level)
    assert (sim["on_hand"] >= 0).all() and (sim["lost"] >= 0).all()
    # pipeline is seeded with L+1 weekly deliveries; an order placed at the end of week t
    # lands at the start of week t+1+L, so only orders from weeks 0..T-2-L arrive in-horizon
    arrivals = init_level * (L + 1) + sim["orders"][:, : T - 1 - L].sum(1)
    sold = d.sum(1) - sim["lost"].sum(1)
    assert np.allclose(S0 + arrivals - sold, sim["on_hand"][:, -1])


def test_orders_respect_case_pack_and_moq():
    d = _demand()
    sim = simulate_sku(d, np.full(40, 150.0), lead_time=1, case_pack=12, moq=36, init_level=50)
    q = sim["orders"][sim["orders"] > 0]
    assert (q >= 36).all()
    assert np.allclose(q % 12, 0)


def test_more_safety_stock_means_fewer_stockouts_and_more_inventory():
    d = _demand(mu=60, P=500)
    lo = simulate_sku(d, np.full(40, 60 * 3 + 0.0), 2, 1, 1, 60)
    hi = simulate_sku(d, np.full(40, 60 * 3 + 120.0), 2, 1, 1, 60)
    assert hi["lost"].sum() < lo["lost"].sum()
    assert hi["on_hand"].mean() > lo["on_hand"].mean()


def test_huge_order_up_to_never_stocks_out():
    d = _demand()
    sim = simulate_sku(d, np.full(40, 1e6), 2, 1, 1, 50)
    assert sim["lost"].sum() == 0


def test_policy_level_formulas():
    f = np.array([10.0, 20.0])
    assert np.allclose(baseline_levels(f, lead_time=2, buffer_weeks=2), f * 5)
    assert np.allclose(supplypulse_levels(f, 2, z=1.5, sigma_cover=10), f * 3 + 15)


def test_normal_loss_known_values():
    assert np.isclose(normal_loss(0.0), 0.3989422804, atol=1e-9)
    assert np.isclose(normal_loss(1.0), 0.0833154705, atol=1e-9)
    assert normal_loss(3.0) < 1e-3


@pytest.fixture
def opt_df():
    rng = np.random.default_rng(3)
    n = 25
    cost = rng.uniform(2, 40, n)
    return pd.DataFrame({
        "sku_id": [f"S{i}" for i in range(n)], "mean_weekly": rng.uniform(20, 300, n),
        "sigma_cover": rng.uniform(10, 150, n), "unit_cost": cost,
        "holding_rate": 0.25, "margin": cost * rng.uniform(0.3, 1.0, n),
    })


def test_optimizer_meets_fill_constraints(opt_df):
    r = optimise_safety_stock(opt_df, target_fill=0.97, min_fill=0.9, penalty_weight=0.0)
    assert r.status.iloc[0] == "Optimal"
    assert 1 - r.exp_short_per_week.sum() / r.mean_weekly.sum() >= 0.97 - 1e-9
    assert (r.fill_rate >= 0.9 - 1e-9).all()
    assert set(r.z).issubset(set(Z_GRID))


def test_optimizer_cost_rises_with_service_target(opt_df):
    costs = [optimise_safety_stock(opt_df, t, 0.9, penalty_weight=0.0).eval("safety_stock * unit_cost").sum()
             for t in (0.93, 0.97, 0.995)]
    assert costs[0] <= costs[1] <= costs[2]


def test_optimizer_beats_uniform_z_at_equal_service(opt_df):
    """Allocating safety stock by volatility and cost must be no worse than one z for all."""
    opt = optimise_safety_stock(opt_df, 0.97, 0.9, penalty_weight=0.0)
    fill_opt = 1 - opt.exp_short_per_week.sum() / opt.mean_weekly.sum()
    for z in Z_GRID:
        short = (opt_df.sigma_cover * normal_loss(z)).sum()
        if 1 - short / opt_df.mean_weekly.sum() >= fill_opt:
            uniform_value = (opt_df.sigma_cover * z * opt_df.unit_cost).sum()
            break
    assert (opt.safety_stock * opt.unit_cost).sum() <= uniform_value + 1e-6
