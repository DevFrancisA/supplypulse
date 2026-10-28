"""Demand forecasting: gradient-boosted model vs. moving-average baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .config import Config
from .features import CATEGORICAL, FEATURES, build_features


def wape(actual, pred) -> float:
    """Weighted absolute percentage error: sum|err| / sum(actual)."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / actual.sum())


def bias(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float((pred - actual).sum() / actual.sum())


def fit_model(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",            # counts, multiplicative structure
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=24,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=CATEGORICAL,
        random_state=seed,
    )
    return model.fit(train[FEATURES], train["units"])


def run_backtest(feats: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, object]:
    """Two expanding-window folds: `calibration` (used only to estimate forecast
    error for safety stock) and `test` (reported accuracy + simulation)."""
    folds = {
        "calibration": (cfg.calib_start, cfg.calib_start, cfg.test_start),
        "test": (cfg.test_start, cfg.test_start, cfg.n_weeks),
    }
    parts, test_model, test_df = [], None, None
    for fold, (train_end, lo, hi) in folds.items():
        train = feats[(feats.week_idx >= cfg.warmup_weeks) & (feats.week_idx < train_end)]
        test = feats[(feats.week_idx >= lo) & (feats.week_idx < hi)].copy()
        model = fit_model(train)
        test["fc_gbm"] = model.predict(test[FEATURES])
        test["fc_ma4"] = test["ma_4"]
        test["fc_snaive"] = test["lag_52"]
        test["fold"] = fold
        parts.append(test)
        if fold == "test":
            test_model, test_df = model, test

    cols = ["sku_id", "category", "week_idx", "week_start", "fold", "units",
            "promo_flag", "fc_gbm", "fc_ma4", "fc_snaive"]
    return pd.concat(parts)[cols].rename(columns={"units": "actual"}).reset_index(drop=True), (test_model, test_df)


def summarise_accuracy(preds: pd.DataFrame) -> dict:
    t = preds[preds.fold == "test"]
    out = {
        "wape_gbm": wape(t.actual, t.fc_gbm),
        "wape_ma4": wape(t.actual, t.fc_ma4),
        "wape_snaive": wape(t.actual.where(t.fc_snaive.notna()), t.fc_snaive.fillna(t.fc_ma4)),
        "bias_gbm": bias(t.actual, t.fc_gbm),
        "bias_ma4": bias(t.actual, t.fc_ma4),
    }
    out["wape_improvement_pct"] = 100 * (1 - out["wape_gbm"] / out["wape_ma4"])
    promo = t[t.promo_flag == 1]
    out["wape_gbm_promo_weeks"] = wape(promo.actual, promo.fc_gbm)
    out["wape_ma4_promo_weeks"] = wape(promo.actual, promo.fc_ma4)
    return out


def accuracy_by(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    t = preds[preds.fold == "test"]
    rows = []
    for k, g in t.groupby(key):
        rows.append({key: k, "volume": g.actual.sum(),
                     "wape_gbm": wape(g.actual, g.fc_gbm),
                     "wape_ma4": wape(g.actual, g.fc_ma4)})
    df = pd.DataFrame(rows)
    df["improvement_pct"] = 100 * (1 - df.wape_gbm / df.wape_ma4)
    return df


def feature_importance(model, test_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    res = permutation_importance(model, test_df[FEATURES], test_df["units"],
                                 n_repeats=3, random_state=seed,
                                 scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": FEATURES, "importance": res.importances_mean})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def forecast_future(model, sales: pd.DataFrame, skus: pd.DataFrame, horizon: int,
                    planned_promos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recursive multi-step forecast: each predicted week is appended to the
    history so the next step's lag/rolling features can use it.

    `planned_promos` (optional) has columns sku_id, h, discount_pct (h = 1..horizon).
    """
    s = sales.copy()
    last = int(s.week_idx.max())
    last_date = pd.to_datetime(s.loc[s.week_idx == last, "week_start"].iloc[0])
    list_price = skus.set_index("sku_id")["unit_price"]
    promos = {}
    if planned_promos is not None:
        promos = {(r.sku_id, int(r.h)): float(r.discount_pct) for r in planned_promos.itertuples()}

    out = []
    for h in range(1, horizon + 1):
        w = last + h
        disc = np.array([promos.get((sid, h), 0.0) for sid in list_price.index])
        new = pd.DataFrame({
            "sku_id": list_price.index, "week_idx": w,
            "week_start": (last_date + pd.Timedelta(weeks=h)).strftime("%Y-%m-%d"),
            "units": 0.0, "price": (list_price.values * (1 - disc)).round(2),
            "promo_flag": (disc > 0).astype(int), "discount_pct": disc,
        })
        feats = build_features(pd.concat([s, new], ignore_index=True), skus)
        row = feats[feats.week_idx == w].set_index("sku_id").loc[list_price.index]
        pred = np.clip(model.predict(row[FEATURES]), 0, None)
        new["units"] = pred
        s = pd.concat([s, new], ignore_index=True)
        out.append(pd.DataFrame({"sku_id": list_price.index, "h": h, "week_idx": w,
                                 "week_start": new["week_start"].values, "forecast": pred}))
    return pd.concat(out, ignore_index=True)
