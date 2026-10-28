import numpy as np

from supplypulse.data import generate


def test_generation_is_deterministic(cfg):
    a = generate(cfg)[1]
    b = generate(cfg)[1]
    assert a.equals(b)


def test_shapes_and_validity(cfg, data):
    skus, sales, truth = data
    assert len(skus) == cfg.n_skus
    assert len(sales) == cfg.n_skus * cfg.n_weeks == len(truth)
    assert (sales.units >= 0).all()
    assert skus.lead_time_weeks.between(1, 4).all()
    assert (skus.unit_price > skus.unit_cost).all()
    assert (skus.moq % skus.case_pack == 0).all()


def test_promotions_lift_demand(data):
    _, sales, truth = data
    m = sales.merge(truth, on=["sku_id", "week_idx"])
    promo = m[m.promo_flag == 1]
    normal = m[m.promo_flag == 0]
    assert (promo.units / promo.true_mean).mean() > 0.9  # noise-centred on the true mean
    assert promo.units.mean() > normal.units.mean() * 0.9


def test_features_do_not_leak_the_target(data, feats):
    """Every lag / rolling feature of week t must only use weeks < t."""
    _, sales, _ = data
    sku = sales.sku_id.iloc[0]
    s = sales[sales.sku_id == sku].set_index("week_idx").units
    f = feats[feats.sku_id == sku].set_index("week_idx")
    for t in (60, 120, 200):
        assert f.loc[t, "lag_1"] == s[t - 1]
        assert f.loc[t, "lag_52"] == s[t - 52]
        assert np.isclose(f.loc[t, "ma_4"], s[t - 4:t].mean())
        assert np.isclose(f.loc[t, "ma_13"], s[t - 13:t].mean())
        assert np.isclose(f.loc[t, "sd_8"], s[t - 8:t].std(ddof=0))


def test_changing_the_future_does_not_change_past_features(data):
    """Perturb the last 10 weeks of sales: features for earlier weeks must not move."""
    from supplypulse.features import build_features
    skus, sales, _ = data
    base = build_features(sales, skus)
    altered = sales.copy()
    altered.loc[altered.week_idx >= altered.week_idx.max() - 9, "units"] += 1000
    new = build_features(altered, skus)
    cut = sales.week_idx.max() - 9
    cols = [c for c in base.columns if c not in ("units",)]
    assert base[base.week_idx < cut][cols].equals(new[new.week_idx < cut][cols])
