"""Missing-value audit and Phase-I-only preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def audit(X: pd.DataFrame) -> dict:
    tr = X.iloc[slice(*C.TRAIN)]
    miss_all = X.isna().mean()
    miss_tr = tr.isna().mean()
    const_all = X.nunique(dropna=True) <= 1
    const_tr = tr.nunique(dropna=True) <= 1
    bins = [0, 1e-12, 0.05, 0.2, 0.5, 1.0001]
    labels = ["0%", "(0,5%]", "(5,20%]", "(20,50%]", ">50%"]
    hist = pd.cut(miss_all, bins=bins, labels=labels, right=False if False else True, include_lowest=True).value_counts().reindex(labels)
    return {
        "cells_missing_frac_all": float(X.isna().to_numpy().mean()),
        "sensors_with_any_missing_all": int((miss_all > 0).sum()),
        "sensors_missing_hist_all": {k: int(v) for k, v in hist.items()},
        "sensors_gt50pct_missing_train": int((miss_tr > C.MAX_MISSING).sum()),
        "sensors_constant_all": int(const_all.sum()),
        "sensors_constant_train": int(const_tr.sum()),
        "per_sensor_missing_pct_all": {k: round(float(v) * 100, 2) for k, v in miss_all.items()},
    }


class Prep:
    """Fit on Phase I-fit passing runs only (missingness filter uses training-period features only)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        tr = X.iloc[slice(*C.TRAIN)]
        keep = tr.columns[tr.isna().mean() <= C.MAX_MISSING]
        self.dropped_missing = [c for c in X.columns if c not in set(keep)]
        fitp = X.iloc[slice(*C.FIT)][y[slice(*C.FIT)] == 0][keep]
        self.median = fitp.median()
        # a column entirely missing among fit passes -> median NaN -> treat as constant (dropped)
        imp = fitp.fillna(self.median)
        sd = imp.std(ddof=1)
        ok = sd.notna() & (sd > 0)
        self.dropped_constant = list(sd.index[~ok])
        self.cols = list(sd.index[ok])
        self.median = self.median[self.cols]
        self.mean = imp[self.cols].mean()
        self.std = sd[self.cols]
        return self

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols].fillna(self.median)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((self.impute(X) - self.mean) / self.std).to_numpy()
