import numpy as np

from supplypulse.forecast import (bias, fit_model, forecast_future, run_backtest,
                                  summarise_accuracy, wape)


def test_wape_and_bias():
    a = np.array([10, 20, 30.0])
    assert wape(a, a) == 0
    assert np.isclose(wape(a, a + 6), 18 / 60)
    assert np.isclose(bias(a, a * 1.1), 0.1)


def test_gbm_beats_moving_average_out_of_sample(cfg, feats):
    preds, _ = run_backtest(feats, cfg)
    acc = summarise_accuracy(preds)
    assert acc["wape_gbm"] < acc["wape_ma4"]
    assert acc["wape_gbm_promo_weeks"] < acc["wape_ma4_promo_weeks"]
    # train / test folds must not overlap in time
    assert preds[preds.fold == "test"].week_idx.min() == cfg.test_start
    assert preds[preds.fold == "calibration"].week_idx.max() < cfg.test_start


def test_recursive_forecast_shape_and_sanity(cfg, data, feats):
    skus, sales, _ = data
    model = fit_model(feats[feats.week_idx >= cfg.warmup_weeks])
    fut = forecast_future(model, sales, skus, horizon=3)
    assert len(fut) == len(skus) * 3
    assert (fut.forecast >= 0).all()
    assert fut.week_idx.min() == cfg.n_weeks and fut.week_idx.max() == cfg.n_weeks + 2
    recent = sales[sales.week_idx >= cfg.n_weeks - 13].groupby("sku_id").units.mean()
    ratio = fut[fut.h == 1].set_index("sku_id").forecast / recent
    assert ratio.between(0.3, 3).all()


def test_planned_promo_raises_forecast(cfg, data, feats):
    import pandas as pd
    skus, sales, _ = data
    model = fit_model(feats[feats.week_idx >= cfg.warmup_weeks])
    plain = forecast_future(model, sales, skus, horizon=1).set_index("sku_id").forecast
    promo = pd.DataFrame({"sku_id": skus.sku_id, "h": 1, "discount_pct": 0.3})
    lifted = forecast_future(model, sales, skus, horizon=1, planned_promos=promo).set_index("sku_id").forecast
    assert lifted.sum() > plain.sum()
