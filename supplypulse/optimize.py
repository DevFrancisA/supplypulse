"""Safety-stock optimisation with PuLP.

Each SKU picks one service level (z-score) from a discrete grid. Holding cost of
the safety stock is traded off against the expected cost of lost sales, subject to
a demand-weighted portfolio fill-rate floor and a per-SKU fill-rate floor. This is
a multiple-choice knapsack, solved to optimality with CBC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
from scipy.stats import norm

Z_GRID = np.round(np.arange(0.0, 3.01, 0.1), 2)


def _binary_vars(prob: pulp.LpProblem, n: int, K: int) -> dict:
    """n x K binary decision variables. PuLP >= 4 creates variables on the problem;
    PuLP 2.x/3.x uses the LpVariable constructor."""
    if hasattr(prob, "add_variable_dict"):
        return prob.add_variable_dict("x", (range(n), range(K)), cat="Binary")
    return {(i, k): pulp.LpVariable(f"x_{i}_{k}", cat="Binary") for i in range(n) for k in range(K)}


def _solve(prob: pulp.LpProblem, time_limit: int = 60) -> str:
    """Solve with CBC when bundled (PuLP < 4), otherwise HiGHS (`pip install highspy`)."""
    available = pulp.listSolvers(onlyAvailable=True)
    if "PULP_CBC_CMD" in available:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    elif "HiGHS" in available:
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit)
    else:
        raise RuntimeError("No MILP solver found. Run: pip install highspy")
    result = prob.solve(solver)
    if isinstance(result, int):                      # PuLP < 4 returns a status code
        return pulp.LpStatus[result]
    return result.status.name                        # PuLP >= 4 returns LpSolveStats


def normal_loss(z):
    """Standard normal loss function G(z) = E[(Z - z)+]."""
    z = np.asarray(z, float)
    return norm.pdf(z) - z * (1 - norm.cdf(z))


def optimise_safety_stock(df: pd.DataFrame, target_fill: float, min_fill: float,
                          z_grid: np.ndarray = Z_GRID, penalty_weight: float = 1.0) -> pd.DataFrame:
    """Choose z per SKU.

    penalty_weight scales the lost-sales cost inside the objective. 1.0 is the true
    cost-minimisation (service level falls out of the economics; the fill-rate
    constraints act as floors). 0.0 minimises holding cost alone, so the portfolio
    fill-rate target is what drives the answer -- used to trace the cost/service frontier.

    df columns: sku_id, mean_weekly (demand per review cycle), sigma_cover (std of
    forecast error over the protection interval), unit_cost, holding_rate, margin.
    Returns df plus z, safety_stock, fill_rate, exp_short_per_week, annual_cost, status.
    """
    df = df.reset_index(drop=True).copy()
    n, K = len(df), len(z_grid)
    sigma = df.sigma_cover.to_numpy(float)
    q = df.mean_weekly.to_numpy(float)

    ss = np.outer(sigma, z_grid)                                   # (n, K) units
    short = sigma[:, None] * normal_loss(z_grid)[None, :]          # expected units short / week
    hold = ss * (df.unit_cost * df.holding_rate).to_numpy()[:, None]
    pen = short * 52 * df.margin.to_numpy()[:, None]
    cost = hold + pen
    fill = 1 - short / np.maximum(q[:, None], 1e-9)

    prob = pulp.LpProblem("safety_stock", pulp.LpMinimize)
    x = _binary_vars(prob, n, K)
    objective = hold + penalty_weight * pen
    prob += pulp.lpSum(objective[i, k] * x[i, k] for i in range(n) for k in range(K))
    for i in range(n):
        prob += pulp.lpSum(x[i, k] for k in range(K)) == 1
        ok = [k for k in range(K) if fill[i, k] >= min_fill]
        if not ok:                       # SKU too volatile for the floor: allow only the top level
            ok = [K - 1]
        for k in range(K):
            if k not in ok:
                prob += x[i, k] == 0
    prob += pulp.lpSum(short[i, k] * x[i, k] for i in range(n) for k in range(K)) <= (1 - target_fill) * q.sum()
    status = _solve(prob)
    if status != "Optimal":              # infeasible target: fall back to the highest level everywhere
        pick = np.full(n, K - 1)
    else:
        pick = np.array([max(range(K), key=lambda k: x[i, k].value() or 0) for i in range(n)])

    idx = np.arange(n)
    out = df.copy()
    out["z"] = z_grid[pick]
    out["safety_stock"] = ss[idx, pick]
    out["exp_short_per_week"] = short[idx, pick]
    out["fill_rate"] = fill[idx, pick]
    out["annual_cost"] = cost[idx, pick]
    out["status"] = status
    return out


def cost_service_frontier(df: pd.DataFrame, targets, min_fill: float) -> pd.DataFrame:
    """Re-solve at several portfolio fill targets (minimum holding cost for each) to
    trace the cost / service trade-off."""
    rows = []
    for tgt in targets:
        r = optimise_safety_stock(df, tgt, min_fill, penalty_weight=0.0)
        rows.append({
            "target_fill_rate": tgt,
            "achieved_fill_rate": 1 - r.exp_short_per_week.sum() / r.mean_weekly.sum(),
            "safety_stock_value": float((r.safety_stock * r.unit_cost).sum()),
            "annual_holding_cost": float((r.safety_stock * r.unit_cost * r.holding_rate).sum()),
            "annual_shortage_cost": float((r.exp_short_per_week * 52 * r.margin).sum()),
            "annual_cost": float(r.annual_cost.sum()),
            "status": r.status.iloc[0],
        })
    return pd.DataFrame(rows)
