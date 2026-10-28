"""Central configuration. Every tunable lives here so runs are reproducible."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    # --- synthetic data -------------------------------------------------
    seed: int = 42
    n_skus: int = 60
    n_weeks: int = 260            # 5 years of weekly history
    start_date: str = "2021-01-04"  # a Monday

    # --- forecasting ----------------------------------------------------
    warmup_weeks: int = 52        # need a year of history for lag_52
    calib_weeks: int = 52         # fold used to estimate forecast error (sigma)
    test_weeks: int = 52          # held-out fold: forecast accuracy + simulation
    horizon_weeks: int = 6        # recursive forecast horizon for recommendations

    # --- inventory policy / simulation ----------------------------------
    n_paths: int = 300            # Monte Carlo demand paths per SKU
    burn_in_weeks: int = 4        # excluded from simulation KPIs
    baseline_buffer_weeks: float = 2.0  # rule-of-thumb: flat 2 weeks of safety cover
    holding_rate: float = 0.25    # annual holding cost, share of unit cost

    # --- optimisation ---------------------------------------------------
    target_fill_rate: float = 0.98   # portfolio (demand-weighted) fill-rate floor
    min_sku_fill_rate: float = 0.90  # per-SKU floor so no SKU is starved

    @property
    def test_start(self) -> int:
        return self.n_weeks - self.test_weeks

    @property
    def calib_start(self) -> int:
        return self.test_start - self.calib_weeks


CONFIG = Config()
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
POWERBI_DIR = ROOT / "powerbi" / "data"
SQL_DIR = Path(__file__).resolve().parent / "sql"
