"""Run all preregistered methods; write results/results.json, results/scores.csv, contributions."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import sklearn

from . import config as C
from .data import ROOT, load
from .methods import (PCAMonitor, autoencoder, gradient_boosting, imr_rules, isolation_forest, rank_sensors)
from .metrics import evaluate, trivial_costs
from .preprocess import Prep, audit

RES = ROOT / "results"


def main():
    RES.mkdir(exist_ok=True)
    X, y, t, order = load()
    assert len(y) == C.N_TOTAL
    fit, cal, test = (slice(*C.FIT), slice(*C.CAL), slice(*C.TEST))
    pos = np.arange(len(y))
    block = np.where(pos < C.FIT[1], "phase1_fit", np.where(pos < C.CAL[1], "phase1_cal", "test"))

    # ---- missing values
    aud = audit(X)
    pd.Series(aud.pop("per_sensor_missing_pct_all"), name="missing_pct_all").rename_axis("sensor").to_csv(RES / "missing_by_sensor.csv")

    prep = Prep().fit(X, y)
    Z = prep.transform(X)
    Ximp = prep.impute(X).to_numpy()
    fit_pass = (pos < C.FIT[1]) & (y == 0)
    cal_pass = (pos >= C.CAL[0]) & (pos < C.CAL[1]) & (y == 0)
    yt = y[test]

    def thr(score, alpha=C.ALPHA):
        return float(np.quantile(score[cal_pass], 1 - alpha))

    scores, methods, secondary, thresholds = {}, {}, {}, {}

    # ---- PCA T2 / SPE
    pca = PCAMonitor().fit(Z[fit_pass])
    t2, spe = pca.scores(Z)
    scores["PCA_T2"], scores["PCA_SPE"] = t2, spe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ---- ML
        if_score = isolation_forest(Z[fit_pass])(Z)
        ae_fn, ae_model = autoencoder(Z[fit_pass])
        ae_score = ae_fn(Z)
        gb_score = gradient_boosting(Z[fit], y[fit])(Z)
    scores["IsolationForest"], scores["Autoencoder"], scores["GradBoost_SUPERVISED"] = if_score, ae_score, gb_score

    for name, s in scores.items():
        th = thr(s)
        thresholds[name] = {"alpha_0.01": th, "alpha_0.0027": thr(s, C.ALPHA_SECONDARY)}
        methods[name] = evaluate(s[test] > th, yt, s[test])
        secondary[f"{name} @alpha=0.27%"] = evaluate(s[test] > thresholds[name]["alpha_0.0027"], yt, s[test])
    pt2, pq = pca.parametric_limits(C.ALPHA)
    thresholds["PCA_T2"]["parametric_F_0.01"] = pt2
    thresholds["PCA_SPE"]["parametric_JM_0.01"] = pq
    secondary["PCA_T2 @parametric F 99%"] = evaluate(t2[test] > pt2, yt, t2[test])
    secondary["PCA_SPE @parametric Jackson-Mudholkar 99%"] = evaluate(spe[test] > pq, yt, spe[test])
    # combined T2 OR SPE (common practice, reported as secondary)
    secondary["PCA_T2_or_SPE @alpha=1% each"] = evaluate(
        (t2[test] > thresholds["PCA_T2"]["alpha_0.01"]) | (spe[test] > thresholds["PCA_SPE"]["alpha_0.01"]), yt)

    # ---- I-MR on top sensors
    top, top_s = rank_sensors(Ximp[fit], y[fit], prep.cols, C.IMR_TOP_K)
    any_we = np.zeros(len(y), bool)
    any_r1 = np.zeros(len(y), bool)
    imr_detail = []
    imr_series = {}
    for c in top:
        j = prep.cols.index(c)
        x = Ximp[:, j]
        xp = x[fit_pass]
        cl = float(xp.mean())
        mrbar = float(np.mean(np.abs(np.diff(xp))))
        sigma = mrbar / 1.128
        r = imr_rules(x, cl, sigma, mrbar)
        we = r["r1"] | r["r2"] | r["r3"] | r["r4"] | r["mr"]
        any_we |= we
        any_r1 |= r["r1"]
        imr_detail.append({
            "sensor": c, "cl": cl, "sigma": sigma, "mrbar": mrbar,
            "test_runs_flagged_by_rule": {k: int(r[k][test].sum()) for k in ["r1", "r2", "r3", "r4", "mr"]},
        })
        imr_series[f"{c}__value"] = x
        imr_series[f"{c}__z"] = r["z"]
    methods["IMR_WE_rules_top10"] = evaluate(any_we[test], yt)
    methods["IMR_rule1_only_top10"] = evaluate(any_r1[test], yt)
    pd.DataFrame({"run_pos": pos, **imr_series}).to_csv(RES / "imr_top_sensors.csv", index=False)

    # ---- contributions for every test alarm from T2 / SPE (alpha=1%)
    rows = []
    for chart, s in [("T2", t2), ("SPE", spe)]:
        th = thresholds[f"PCA_{chart}"]["alpha_0.01"]
        for i in np.where((pos >= C.TEST[0]) & (s > th))[0]:
            c_t2, c_spe = pca.contributions(Z[i])
            c = c_t2 if chart == "T2" else c_spe
            tot = float(np.sum(c))
            for rnk, j in enumerate(np.argsort(-c)[:15]):
                rows.append({"run_pos": int(i), "timestamp": str(t[i]), "label": "fail" if y[i] else "pass",
                             "chart": chart, "score": float(s[i]), "limit": th, "rank": rnk + 1,
                             "sensor": prep.cols[j], "contribution": float(c[j]), "share_of_total": float(c[j]) / tot})
    pd.DataFrame(rows).to_csv(RES / "contributions_test_alarms.csv", index=False)

    # ---- scores table
    sc = pd.DataFrame({"run_pos": pos, "timestamp": t.astype(str), "label": np.where(y == 1, "fail", "pass"), "block": block})
    for k, v in scores.items():
        sc[k] = v
    sc["IMR_WE_alarm"] = any_we.astype(int)
    sc["IMR_rule1_alarm"] = any_r1.astype(int)
    sc.to_csv(RES / "scores.csv", index=False)

    def counts(sl):
        return {"n": int(len(y[sl])), "fails": int(y[sl].sum()), "passes": int((y[sl] == 0).sum()),
                "first_run": int(pos[sl][0]), "last_run": int(pos[sl][-1]),
                "time_start": str(t[sl].iloc[0]), "time_end": str(t[sl].iloc[-1])}

    out = {
        "generated_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
        "time_order": {"already_sorted_in_file": bool((order == np.arange(len(order))).all()),
                       "rows_with_tied_timestamp": int(t.duplicated().sum()), "tie_rule": "stable sort, file order kept"},
        "splits": {"phase1_fit": counts(fit), "phase1_cal": counts(cal), "test": counts(test)},
        "missing_audit": aud,
        "preprocessing": {"sensors_raw": X.shape[1], "dropped_gt50pct_missing_train": len(prep.dropped_missing),
                          "dropped_constant_phase1_fit_pass": len(prep.dropped_constant), "sensors_kept": len(prep.cols),
                          "imputation": "median of Phase I-fit passing runs", "scaling": "z-score, Phase I-fit passing runs",
                          "dropped_missing_list": prep.dropped_missing},
        "pca": {"k": pca.k, "var_explained": pca.var_explained, "n_fit": int(pca.n)},
        "imr": {"top_sensors": top, "rank_score_abs_auc_minus_half": top_s, "detail": imr_detail},
        "autoencoder": {"n_iter": int(ae_model.n_iter_), "final_loss": float(ae_model.loss_)},
        "thresholds": thresholds,
        "methods_primary_alpha_0.01": methods,
        "secondary": secondary,
        "trivial_policies_cost": trivial_costs(yt),
    }
    (RES / "results.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    for k, m in methods.items():
        print(f"{k:24s} DR={m['detection_rate']:.3f} FAR={m['false_alarm_rate']:.3f} cost@R10={m['cost_primary']:.4f}")


if __name__ == "__main__":
    main()
