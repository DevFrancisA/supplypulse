import numpy as np
import pandas as pd

from supplypulse.whatif import scenario, totals


def _rec():
    n = 8
    rng = np.random.default_rng(1)
    L = rng.integers(1, 5, n)
    weekly = rng.uniform(30, 120, n)
    return pd.DataFrame({
        "sku_id": [f"S{i}" for i in range(n)], "category": "Snacks", "abc_class": "A",
        "unit_cost": rng.uniform(2, 20, n), "lead_time_weeks": L, "case_pack": 12, "moq": 24,
        "sigma_cover": rng.uniform(20, 90, n), "z": rng.uniform(1, 2.5, n),
        "on_hand": rng.uniform(20, 300, n), "on_order": rng.uniform(0, 200, n),
        "cover_demand": weekly * (L + 1),
    })


def test_base_scenario_is_order_up_to_formula():
    rec = _rec()
    s = scenario(rec)
    expected = rec.cover_demand + rec.z * rec.sigma_cover
    assert np.allclose(s.order_up_to, expected)
    assert (s.order_qty % 12 == 0).all()
    assert (s.order_qty[s.order_qty > 0] >= 24).all()


def test_scenarios_move_in_the_expected_direction():
    rec = _rec()
    base = totals(scenario(rec))
    assert totals(scenario(rec, delay_weeks=2))["ss_value"] > base["ss_value"]
    assert totals(scenario(rec, uplift=0.3))["order_value"] >= base["order_value"]
    assert totals(scenario(rec, optimised=False, service_level=0.999))["expected_fill"] > \
           totals(scenario(rec, optimised=False, service_level=0.80))["expected_fill"]
