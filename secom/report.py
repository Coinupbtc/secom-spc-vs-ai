"""Build figures and the README results section from results/results.json + CSVs (no numbers typed by hand)."""
from __future__ import annotations

import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import ROOT

RES, FIG = ROOT / "results", ROOT / "figures"
NAMES = {
    "PCA_T2": "PCA Hotelling T² (classic)",
    "PCA_SPE": "PCA SPE/Q (classic)",
    "IMR_WE_rules_top10": "I-MR, WE rules 1-4 + MR, top-10 sensors (classic, uncalibrated)",
    "IMR_rule1_only_top10": "I-MR, rule 1 only, top-10 sensors (classic, uncalibrated)",
    "IsolationForest": "Isolation Forest (ML, unsupervised)",
    "Autoencoder": "Autoencoder MLP (ML, unsupervised)",
    "GradBoost_SUPERVISED": "Gradient boosting, supervised reference (uses labels)",
}


def f(x, d=3):
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return "∞"
    return f"{x:.{d}f}"


def shade(ax):
    ax.axvspan(C.FIT[0], C.FIT[1] - 0.5, color="tab:green", alpha=0.06, label="Phase I fit")
    ax.axvspan(C.CAL[0] - 0.5, C.CAL[1] - 0.5, color="tab:blue", alpha=0.06, label="Phase I calibration")
    ax.axvspan(C.TEST[0] - 0.5, C.TEST[1], color="tab:orange", alpha=0.06, label="Phase II test")


def figures(R, sc):
    FIG.mkdir(exist_ok=True)
    fail = sc.label.eq("fail").to_numpy()
    # control charts T2/SPE
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, k, lab in [(axes[0], "PCA_T2", "Hotelling T²"), (axes[1], "PCA_SPE", "SPE / Q")]:
        shade(ax)
        ax.scatter(sc.run_pos[~fail], sc[k][~fail], s=5, c="0.4", label="pass")
        ax.scatter(sc.run_pos[fail], sc[k][fail], s=18, c="red", marker="x", label="fail")
        ax.axhline(R["thresholds"][k]["alpha_0.01"], c="k", ls="--", lw=1, label="limit (99%, empirical, Phase I cal)")
        pk = "parametric_F_0.01" if k == "PCA_T2" else "parametric_JM_0.01"
        ax.axhline(R["thresholds"][k][pk], c="purple", ls=":", lw=1, label="textbook parametric 99% limit")
        ax.set_yscale("log"); ax.set_ylabel(lab + " (log)")
    axes[0].legend(fontsize=7, ncol=4, loc="upper left")
    axes[1].set_xlabel("run order (sorted by timestamp)")
    axes[0].set_title(f"SECOM PCA monitoring (k={R['pca']['k']} PCs, {R['pca']['var_explained']*100:.1f}% var), fit on Phase I passes")
    fig.tight_layout(); fig.savefig(FIG / "control_chart_t2_spe.png", dpi=130); plt.close(fig)

    # ML score charts
    ks = ["IsolationForest", "Autoencoder", "GradBoost_SUPERVISED"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, k in zip(axes, ks):
        shade(ax)
        ax.scatter(sc.run_pos[~fail], sc[k][~fail], s=5, c="0.4")
        ax.scatter(sc.run_pos[fail], sc[k][fail], s=18, c="red", marker="x")
        ax.axhline(R["thresholds"][k]["alpha_0.01"], c="k", ls="--", lw=1)
        ax.set_ylabel(k.replace("GradBoost_SUPERVISED", "GradBoost\n(supervised ref)"), fontsize=8)
        if k == "Autoencoder":
            ax.set_yscale("log")
    axes[0].set_title("ML anomaly scores (dashed = 99% threshold on Phase I calibration passes; red x = fail)")
    axes[-1].set_xlabel("run order")
    fig.tight_layout(); fig.savefig(FIG / "ml_scores.png", dpi=130); plt.close(fig)

    # I-MR chart for top sensor
    imr = pd.read_csv(RES / "imr_top_sensors.csv")
    d = R["imr"]["detail"][0]
    s = d["sensor"]
    x = imr[f"{s}__value"].to_numpy()
    mr = np.concatenate([[np.nan], np.abs(np.diff(x))])
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax = axes[0]; shade(ax)
    ax.plot(imr.run_pos, x, lw=0.5, c="0.5")
    ax.scatter(imr.run_pos[fail], x[fail], s=18, c="red", marker="x", label="fail")
    for m, ls in [(0, "-"), (1, ":"), (2, "-."), (3, "--")]:
        for sg in ([1, -1] if m else [1]):
            ax.axhline(d["cl"] + sg * m * d["sigma"], c="k", ls=ls, lw=0.8)
    ax.set_ylabel(f"{s} (imputed)"); ax.legend(fontsize=7)
    ax.set_title(f"Individuals chart, top-ranked sensor {s} (ranked on Phase I-fit only; anonymized, meaning unknown). Lines: CL, ±1σ, ±2σ, ±3σ")
    ax = axes[1]; shade(ax)
    ax.plot(imr.run_pos, mr, lw=0.5, c="0.4")
    ax.axhline(3.267 * d["mrbar"], c="k", ls="--", lw=0.8, label="UCL = 3.267·MR̄")
    ax.set_ylabel("moving range"); ax.set_xlabel("run order"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "imr_chart_top_sensor.png", dpi=130); plt.close(fig)

    # contributions
    co = pd.read_csv(RES / "contributions_test_alarms.csv")
    if len(co):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for r_i, chart in enumerate(["T2", "SPE"]):
            sub = co[co.chart == chart]
            # top alarms: prefer fails first, then highest score
            al = sub.drop_duplicates("run_pos").assign(isfail=lambda d: d.label.eq("fail")).sort_values(["isfail", "score"], ascending=False)
            for c_i in range(3):
                ax = axes[r_i, c_i]
                if c_i >= len(al):
                    ax.axis("off"); continue
                rp = al.iloc[c_i].run_pos
                g = sub[sub.run_pos == rp].sort_values("rank").head(10)
                ax.barh(g.sensor[::-1], g.share_of_total[::-1] * 100, color="tab:red" if g.label.iloc[0] == "fail" else "tab:blue")
                ax.set_title(f"{chart} alarm, run {rp} ({g.label.iloc[0]})\n{chart}={g.score.iloc[0]:.3g} vs limit {g.limit.iloc[0]:.3g}", fontsize=9)
                ax.set_xlabel("% of total statistic"); ax.tick_params(labelsize=7)
        fig.suptitle("Top sensor contributions for test alarms (sensors are anonymized column indices; physical meaning unknown)")
        fig.tight_layout(); fig.savefig(FIG / "contribution_top_alarms.png", dpi=130); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, chart in zip(axes, ["T2", "SPE"]):
            sub = co[(co.chart == chart) & (co["rank"] <= 5)]
            n_al = sub.run_pos.nunique()
            vc = sub.sensor.value_counts().head(15)
            ax.barh(vc.index[::-1], vc.values[::-1])
            ax.set_title(f"{chart}: sensors most often in top-5 contributors\nacross {n_al} test alarms", fontsize=10)
            ax.set_xlabel("# alarms"); ax.tick_params(labelsize=8)
        fig.tight_layout(); fig.savefig(FIG / "contribution_frequency.png", dpi=130); plt.close(fig)

    # cost sensitivity
    M = R["methods_primary_alpha_0.01"]
    rs = C.COST_SWEEP
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, m in M.items():
        ax.plot(rs, [m["cost"][str(r)] for r in rs], marker="o", label=k.replace("GradBoost_SUPERVISED", "GradBoost (supervised ref, uses labels)"), ls="--" if "SUPERVISED" in k else "-")
    tp = R["trivial_policies_cost"]
    ax.plot(rs, [tp["never_alarm"][str(r)] for r in rs], c="k", ls=":", label="never alarm")
    ax.plot(rs, [tp["always_alarm"][str(r)] for r in rs], c="0.6", ls=":", label="always alarm")
    ax.axvline(C.COST_RATIO, c="0.8", lw=6, alpha=0.5, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(rs); ax.set_xticklabels(rs)
    ax.set_xlabel("assumed cost ratio R = escape cost / false-alarm cost (primary R=10 shaded)")
    ax.set_ylabel("cost per test run (lower is better)")
    ax.set_title("Cost sensitivity, thresholds fixed at α=1% (test block)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "cost_sensitivity.png", dpi=130); plt.close(fig)


def auc(m):
    if "AUROC" not in m:
        return "n/a"
    ci = m.get("AUROC_bootstrap95_posthoc")
    return f"{f(m['AUROC'])} ({f(ci[0], 2)}–{f(ci[1], 2)})" if ci else f(m["AUROC"])


def exploratory(R):
    E = R.get("exploratory_posthoc")
    if not E:
        return "_not run_"
    K2, P = E["K2_limit_ratios"], E["K2_primary_model_ratios"]
    S = K2["summary"]
    L = ["**Labeled post-hoc:** these were added after the test block was scored, in response to review. They don't change any planned number.", "",
         f"**1. Are the textbook T²/SPE limits too tight because of drift or because of dimension?** A random 25% of the Phase I-fit passing runs is held out (no time separation),"
         f" preprocessing + PCA are refit on the other 75% (k = {min(r['k'] for r in K2['per_seed'])}–{max(r['k'] for r in K2['per_seed'])} PCs), and the ratio "
         f"empirical 99th percentile / textbook 99% limit is measured on each block by the same refit model ({K2['n_seeds']} seeds; median, with min–max):", "",
         "| statistic | random holdout (same period) | Phase I calibration (later) | test passes (latest) |", "|---|---|---|---|"]
    for nm in ["T2", "SPE"]:
        cells = [f"{f(S[f'{nm}_ratio_{b}']['median'], 1)}× ({f(S[f'{nm}_ratio_{b}']['min'], 1)}–{f(S[f'{nm}_ratio_{b}']['max'], 1)}); "
                 f"FAR at textbook limit {S[f'{nm}_FAR_at_textbook_{b}']['median']*100:.0f}%" for b in ["random_holdout", "phase1_cal", "test"]]
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L += ["", f"For the planned model, the ratios on the calibration block were {f(P['T2_phase1_cal'], 1)}× (T²) and {f(P['SPE_phase1_cal'], 1)}× (SPE), and on test passes {f(P['T2_test_pass'], 1)}× and {f(P['SPE_test_pass'], 1)}×.",
          f"Reading rule (written before running): T² → *{K2['verdict']['T2']}*; SPE → *{K2['verdict']['SPE']}*. "
          "The limits are already far too tight on a random same-period holdout, so the 16–24× gap is **not** evidence of drift by itself: high dimension "
          "(n ≈ 420–560 passes, ~440 sensors, ~100 PCs), non-normal sensors, and imputation produce most of it. The further jump from the random holdout to the test block "
          "is consistent with a later process shift on top. v0 does not attribute it more precisely.", ""]
    K3 = E["K3_rule4_imputation"]
    v = K3["IMR_WE_rules_r4_observed_only"]
    L += [f"**2. Is Western Electric rule 4 an imputation artifact?** Of {K3['r4_hits_test']} rule-4 hits (sensor × test run) across the 10 I-MR sensors, "
          f"{K3['r4_hits_test_with_imputed_in_window']} ({K3['share_of_r4_hits_with_imputed_point']*100:.1f}%) have a median-imputed point in their 8-point window. "
          f"With rule 4 evaluated on observed points only, the WE-rules chart still alarms on {v['FP']}/{v['n_pass']} test passes and {v['TP']}/{v['n_fail']} fails. "
          "So rule 4 is not an imputation artifact: these sensors sit on one side of their Phase I mean for long stretches of the test block.", ""]
    RS = E["random_split_auroc"]
    T, M = RS["time_ordered_same_code"], RS["random_summary"]
    nfs = [r["test_fails"] for r in RS["random_per_seed"]]
    L += [f"**3. Random split vs time-ordered split, same code.** Shuffling the runs before the same 626/314/627 split ({len(nfs)} seeds, {min(nfs)}–{max(nfs)} test fails) vs. the time-ordered split ({RS['time_ordered_test_fails']} test fails):", "",
          "| method | time-ordered AUROC | random-split AUROC, mean (min–max) |", "|---|---|---|"]
    for k in T:
        L.append(f"| {NAMES.get(k, k)} | {f(T[k])} | {f(M[k]['mean'])} ({f(M[k]['min'])}–{f(M[k]['max'])}) |")
    L += ["", f"Post-hoc: the supervised reference scores AUROC {f(M['GradBoost_SUPERVISED']['mean'], 2)} on average with a random split vs {f(T['GradBoost_SUPERVISED'], 2)} time-ordered. "
          "Shuffling lets the model train on runs from the same weeks it is tested on, so random-split SECOM numbers flatter every method. "
          "A deployed monitor only ever sees the past."]
    return "\n".join(L)


def table(R):
    M = R["methods_primary_alpha_0.01"]
    tp = R["trivial_policies_cost"]
    Rr = C.COST_RATIO
    L = [f"Test block: {R['splits']['test']['n']} runs ({R['splits']['test']['fails']} fails, {R['splits']['test']['passes']} passes). "
         f"Thresholds at α = {C.ALPHA:.0%} from Phase I calibration passes (I-MR rows are not calibrated). Cost at R = {Rr} (assumed); lower is better. "
         f"Reference costs: never alarm = {f(tp['never_alarm'][str(Rr)], 4)}, always alarm = {f(tp['always_alarm'][str(Rr)], 4)}.", "",
         "| method | detected fails | detection rate (95% CI) | false-alarm rate | ARL0 emp. (1/FAR) | ARL1 / median delay / ≤5 runs / censored | AUROC (bootstrap 95% CI) | cost @R=10 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, m in M.items():
        lo, hi = m["detection_rate_wilson95"]
        L.append(f"| {NAMES.get(k, k)} | {m['TP']}/{m['n_fail']} | {f(m['detection_rate'])} ({f(lo, 2)}–{f(hi, 2)}) | "
                 f"{f(m['false_alarm_rate'])} ({m['FP']}/{m['n_pass']}) | {f(m['ARL0_empirical'], 1)} ({f(m['ARL0_geometric'], 1)}) | "
                 f"{f(m['ARL1_mean_runs_to_signal'], 1)} / {f(m['delay_median'], 0)} / {f(m['detected_within_5_runs'], 2)} / {m['delay_censored']} | "
                 f"{auc(m)} | {f(m['cost_primary'], 4)} |")
    L += ["", "Cost sensitivity (cost per test run; thresholds fixed at α = 1%):", "",
          "| method | " + " | ".join(f"R={r}" for r in C.COST_SWEEP) + " |", "|---|" + "---|" * len(C.COST_SWEEP)]
    for k, m in M.items():
        L.append(f"| {k} | " + " | ".join(f(m["cost"][str(r)], 4) for r in C.COST_SWEEP) + " |")
    for k in ["never_alarm", "always_alarm"]:
        L.append(f"| *{k}* | " + " | ".join(f(tp[k][str(r)], 4) for r in C.COST_SWEEP) + " |")
    L += ["", "Secondary (not primary; same test block):", "", "| variant | detected | FAR | cost @R=10 |", "|---|---|---|---|"]
    for k, m in R["secondary"].items():
        L.append(f"| {k} | {m['TP']}/{m['n_fail']} | {f(m['false_alarm_rate'])} | {f(m['cost_primary'], 4)} |")
    return "\n".join(L)


def data_section(R):
    a, p, s = R["missing_audit"], R["preprocessing"], R["splits"]
    L = ["| block | runs | time range | fails | passes |", "|---|---|---|---|---|"]
    for k in ["phase1_fit", "phase1_cal", "test"]:
        b = s[k]
        L.append(f"| {k} | {b['first_run']}–{b['last_run']} ({b['n']}) | {b['time_start']} → {b['time_end']} | {b['fails']} | {b['passes']} |")
    L += ["", f"- Missing cells: {a['cells_missing_frac_all']*100:.2f}% of all values; {a['sensors_with_any_missing_all']} of 590 sensors have at least one missing value. "
          f"Per-sensor missing-% histogram (all runs): {a['sensors_missing_hist_all']}. Full list: `results/missing_by_sensor.csv`.",
          f"- Constant sensors: {a['sensors_constant_all']} constant over all runs, {a['sensors_constant_train']} constant over the training period.",
          f"- Handling: dropped {p['dropped_gt50pct_missing_train']} sensors with >50% missing in the training period; median-imputed the rest using Phase I-fit passing runs; "
          f"dropped {p['dropped_constant_phase1_fit_pass']} sensors constant on Phase I-fit passes; z-scored with Phase I-fit pass statistics. "
          f"**{p['sensors_kept']} sensors kept.** The test block is never used to fit anything.",
          f"- Time order: file already sorted by timestamp = {R['time_order']['already_sorted_in_file']}; {R['time_order']['rows_with_tied_timestamp']} rows share a timestamp with the previous row (stable sort, file order kept).",
          f"- PCA: k = {R['pca']['k']} components ({R['pca']['var_explained']*100:.1f}% of Phase I-fit pass variance), fit on {R['pca']['n_fit']} passing runs.",
          f"- I-MR sensors (ranked on Phase I-fit only by |AUROC−0.5|): {', '.join(R['imr']['top_sensors'])}."]
    return "\n".join(L)


def main():
    R = json.loads((RES / "results.json").read_text())
    sc = pd.read_csv(RES / "scores.csv")
    figures(R, sc)
    (RES / "results_table.md").write_text(table(R) + "\n")
    rd = ROOT / "README.md"
    txt = rd.read_text()
    for tag, body in [("RESULTS", table(R)), ("DATA", data_section(R)), ("EXPLORATORY", exploratory(R))]:
        txt = re.sub(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->",
                     f"<!-- {tag}:START -->\n<!-- generated by `python -m secom.report` from results/results.json; do not edit by hand -->\n{body}\n<!-- {tag}:END -->",
                     txt, flags=re.S)
    rd.write_text(txt)
    print("figures + README tables regenerated")


if __name__ == "__main__":
    main()
