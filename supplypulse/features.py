"""Feature engineering. Lags and rolling windows are computed in SQL (window
functions); calendar encodings and ratios are added in pandas."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .config import SQL_DIR

FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_52",
    "ma_4", "ma_8", "ma_13", "sd_8", "ly_avg3", "ratio_4_13",
    "promo_flag", "discount_pct", "promo_lag_1", "promo_rate_13",
    "woy_sin", "woy_cos", "woy", "cat_code",
]
CATEGORICAL = ["cat_code"]


def build_features(sales: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    """Run features.sql over `sales` in an in-memory SQLite database."""
    from .data import to_sqlite  # local import avoids a cycle

    with sqlite3.connect(":memory:") as conn:
        to_sqlite(conn, skus, sales[["sku_id", "week_idx", "week_start", "units",
                                     "price", "promo_flag", "discount_pct"]])
        df = pd.read_sql_query((SQL_DIR / "features.sql").read_text(), conn)

    df["sd_8"] = np.sqrt(np.clip(df["msq_8"] - df["ma_8"] ** 2, 0, None))
    df["ratio_4_13"] = df["ma_4"] / (df["ma_13"] + 1.0)
    woy = pd.to_datetime(df["week_start"]).dt.isocalendar().week.astype(float)
    df["woy"] = woy
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)

    categories = sorted(skus["category"].unique())
    cat = skus.set_index("sku_id")["category"]
    df["category"] = df["sku_id"].map(cat)
    df["cat_code"] = pd.Categorical(df["category"], categories=categories).codes
    return df.drop(columns=["msq_8"])
