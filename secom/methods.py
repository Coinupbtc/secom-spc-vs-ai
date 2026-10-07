"""Detection methods. All unsupervised methods are fit on Phase I-fit PASSING runs only."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor

from . import config as C


class PCAMonitor:
    def fit(self, Z: np.ndarray):
        full = PCA(svd_solver="full").fit(Z)
        cum = np.cumsum(full.explained_variance_ratio_)
        self.k = int(np.searchsorted(cum, C.PCA_VAR) + 1)
        self.var_explained = float(cum[self.k - 1])
        self.mu = full.mean_
        self.P = full.components_[: self.k].T            # p x k
        self.lam = full.explained_variance_[: self.k]
        self.lam_resid = full.explained_variance_[self.k:]
        self.n = Z.shape[0]
        return self

    def scores(self, Z):
        Zc = Z - self.mu
        T = Zc @ self.P
        t2 = np.sum(T ** 2 / self.lam, axis=1)
        E = Zc - T @ self.P.T
        spe = np.sum(E ** 2, axis=1)
        return t2, spe

    def contributions(self, z):
        zc = z - self.mu
        M = self.P @ np.diag(1 / self.lam) @ self.P.T
        c_t2 = zc * (M @ zc)
        e = zc - self.P @ (self.P.T @ zc)
        c_spe = e ** 2
        return c_t2, c_spe

    def parametric_limits(self, alpha):
        n, k = self.n, self.k
        t2 = k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(1 - alpha, k, n - k)
        th1, th2, th3 = (np.sum(self.lam_resid ** i) for i in (1, 2, 3))
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        za = stats.norm.ppf(1 - alpha)
        q = th1 * (za * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
        return float(t2), float(q)


def rank_sensors(Ximp_fit, y_fit, cols, k):
    """Rank by |AUROC-0.5| on Phase I-fit runs (uses Phase I labels only)."""
    s = []
    for j, c in enumerate(cols):
        x = Ximp_fit[:, j]
        if np.std(x) == 0:
            s.append(0.0)
            continue
        s.append(abs(roc_auc_score(y_fit, x) - 0.5))
    order = np.argsort(-np.array(s), kind="mergesort")[:k]
    return [cols[i] for i in order], [float(s[i]) for i in order]


def imr_rules(x: np.ndarray, cl: float, sigma: float, mrbar: float):
    """Western Electric rules on a full run-ordered series. Returns dict of boolean arrays."""
    z = (x - cl) / sigma
    n = len(z)

    def window_count(cond, w):
        c = np.convolve(cond.astype(int), np.ones(w, int), mode="full")[:n]
        return c

    r1 = np.abs(z) > 3
    r2 = (window_count(z > 2, 3) >= 2) | (window_count(z < -2, 3) >= 2)
    r3 = (window_count(z > 1, 5) >= 4) | (window_count(z < -1, 5) >= 4)
    r4 = (window_count(z > 0, 8) >= 8) | (window_count(z < 0, 8) >= 8)
    mr = np.concatenate([[0.0], np.abs(np.diff(x))])
    rmr = mr > 3.267 * mrbar
    return {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "mr": rmr, "z": z, "mr_vals": mr}


def isolation_forest(Zfit):
    m = IsolationForest(n_estimators=500, random_state=C.SEED, n_jobs=4).fit(Zfit)
    return lambda Z: -m.score_samples(Z)


def autoencoder(Zfit):
    m = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", max_iter=500, random_state=C.SEED)
    m.fit(Zfit, Zfit)
    return lambda Z: np.mean((m.predict(Z) - Z) ** 2, axis=1), m


def gradient_boosting(Zfit_all, yfit_all):
    m = HistGradientBoostingClassifier(class_weight="balanced", random_state=C.SEED).fit(Zfit_all, yfit_all)
    return lambda Z: m.predict_proba(Z)[:, 1]
