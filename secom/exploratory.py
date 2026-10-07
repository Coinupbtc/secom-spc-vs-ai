"""EXPLORATORY, POST-HOC analyses (added after the test block was scored, in response to review).
None of this is part of the frozen analysis plan; results are reported separately and labeled post-hoc.

K2  Are the T2/SPE limits too tight because of drift, or because of high dimension / non-normality / imputation?
    Hold out a RANDOM (non-time) 25% slice of the Phase I-fit passing runs, refit preprocessing + PCA on the remaining 75%,
    and compare ratio = empirical 99th percentile / textbook 99% limit on (a) the random holdout, (b) Phase I-calibration passes,
    (c) test passes, all scored by the same refit model. Repeated over 20 random seeds.
    Reading rule, written before running: random-holdout ratio < 2 while later blocks > 5 -> supports drift;
    random-holdout ratio >= 5 -> high dimension / non-normality / imputation is a major cause on its own;
    anything else -> not separated. Caveat: a random holdout from the same period shares time-local structure with the fit runs,
    so it is, if anything, flattering to the model (a large random-holdout ratio is strong evidence; a small one is weaker).

RANDOM-SPLIT AUROC  The same pipeline code, but the 1,567 runs are randomly permuted before the 626/314/627 split
    (no time order), repeated over 10 seeds; compared with the time-ordered split computed by the same function.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .methods import PCAMonitor, autoencoder, gradient_boosting, isolation_forest
from .preprocess import Prep


def k2_limit_ratios(X, y, seeds=range(20), frac=0.25):
    pos = np.arange(len(y))
    fitpass = np.where((pos < C.FIT[1]) & (y == 0))[0]
    calp = (pos >= C.CAL[0]) & (pos < C.CAL[1]) & (y == 0)
    testp = (pos >= C.TEST[0]) & (y == 0)
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(fitpass)
        nh = int(round(frac * len(fitpass)))
        hold = np.zeros(len(y), bool); hold[perm[:nh]] = True
        fitm = np.zeros(len(y), bool); fitm[perm[nh:]] = True
        prep = Prep().fit(X, y, fit_mask=fitm)
        Z = prep.transform(X)
        pca = PCAMonitor().fit(Z[fitm])
        t2, spe = pca.scores(Z)
        pt2, pq = pca.parametric_limits(C.ALPHA)
        r = {"seed": int(s), "n_fit": int(fitm.sum()), "n_holdout": int(nh), "k": pca.k}
        for nm, sc, lim in [("T2", t2, pt2), ("SPE", spe, pq)]:
            for bn, m in [("random_holdout", hold), ("phase1_cal", calp), ("test", testp)]:
                r[f"{nm}_ratio_{bn}"] = float(np.quantile(sc[m], 1 - C.ALPHA) / lim)
                r[f"{nm}_FAR_at_textbook_{bn}"] = float(np.mean(sc[m] > lim))
        rows.append(r)
    keys = [k for k in rows[0] if "ratio" in k or "FAR" in k]
    summ = {k: {"median": float(np.median([r[k] for r in rows])), "min": float(np.min([r[k] for r in rows])),
                "max": float(np.max([r[k] for r in rows]))} for k in keys}

    def verdict(nm):
        rh, rc, rt = (summ[f"{nm}_ratio_{b}"]["median"] for b in ("random_holdout", "phase1_cal", "test"))
        if rh < 2 and min(rc, rt) > 5:
            return "supports drift"
        if rh >= 5:
            return "high dimension / non-normality / imputation is a major cause on its own"
        return "not separated"

    return {"per_seed": rows, "summary": summ, "verdict": {"T2": verdict("T2"), "SPE": verdict("SPE")},
            "n_seeds": len(rows), "holdout_frac": frac}


def pipeline_aurocs(X, y, fit_all, fit_pass, train, test):
    """Same preprocessing + models as secom.run, on arbitrary masks. Returns test AUROC per method."""
    prep = Prep().fit(X, y, fit_mask=fit_pass, train_mask=train)
    Z = prep.transform(X)
    pca = PCAMonitor().fit(Z[fit_pass])
    t2, spe = pca.scores(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = {"PCA_T2": t2, "PCA_SPE": spe, "IsolationForest": isolation_forest(Z[fit_pass])(Z),
              "Autoencoder": autoencoder(Z[fit_pass])[0](Z), "GradBoost_SUPERVISED": gradient_boosting(Z[fit_all], y[fit_all])(Z)}
    return {k: float(roc_auc_score(y[test], v[test])) for k, v in sc.items()}, int(y[test].sum())


def random_split_auroc(X, y, seeds=range(10)):
    n = len(y)

    def masks(order):
        rank = np.empty(n, int); rank[order] = np.arange(n)
        fit_all = rank < C.FIT[1]
        train = rank < C.TRAIN[1]
        test = rank >= C.TEST[0]
        return fit_all, fit_all & (y == 0), train, test

    time_auc, time_nf = pipeline_aurocs(X, y, *masks(np.arange(n)))
    rnd = []
    for s in seeds:
        a, nf = pipeline_aurocs(X, y, *masks(np.random.default_rng(1000 + s).permutation(n)))
        rnd.append({"seed": int(s), "test_fails": nf, **a})
    meth = list(time_auc)
    return {"time_ordered_same_code": time_auc, "time_ordered_test_fails": time_nf, "random_per_seed": rnd,
            "random_summary": {m: {"mean": float(np.mean([r[m] for r in rnd])), "min": float(np.min([r[m] for r in rnd])),
                                   "max": float(np.max([r[m] for r in rnd]))} for m in meth}}
