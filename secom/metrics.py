"""Preregistered evaluation metrics (test block, run order)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(alarm: np.ndarray, y: np.ndarray, score: np.ndarray | None = None) -> dict:
    """alarm, y, score are TEST-block arrays in run order (y: 1 fail, 0 pass)."""
    alarm = alarm.astype(bool)
    n = len(y)
    fails = np.where(y == 1)[0]
    passes = np.where(y == 0)[0]
    tp = int(alarm[fails].sum())
    fp = int(alarm[passes].sum())
    fn = len(fails) - tp
    dr = tp / len(fails)
    far = fp / len(passes)
    # ARL0 empirical: gaps (in pass runs) between consecutive false alarms in the pass sequence
    pa = alarm[passes]
    idx = np.where(pa)[0]
    if len(idx) == 0:
        arl0_emp = float("inf")
    else:
        gaps = np.diff(np.concatenate([[-1], idx]))
        arl0_emp = float(np.mean(gaps))
    # detection delay
    alarm_pos = np.where(alarm)[0]
    delays, censored = [], 0
    for i in fails:
        j = alarm_pos[alarm_pos >= i]
        if len(j):
            delays.append(int(j[0] - i))
        else:
            delays.append(int(n - 1 - i))
            censored += 1
    delays = np.array(delays)
    out = {
        "n_test": n, "n_fail": int(len(fails)), "n_pass": int(len(passes)),
        "TP": tp, "FN": fn, "FP": fp, "TN": int(len(passes) - fp),
        "detection_rate": dr, "detection_rate_wilson95": wilson(tp, len(fails)),
        "false_alarm_rate": far, "false_alarm_rate_wilson95": wilson(fp, len(passes)),
        "ARL0_geometric": (1 / far) if far > 0 else float("inf"),
        "ARL0_empirical": arl0_emp,
        "ARL1_mean_runs_to_signal": float(np.mean(delays + 1)),
        "delay_median": float(np.median(delays)),
        "detected_within_5_runs": float(np.mean(delays <= 5)),
        "delay_censored": censored,
        "cost": {str(r): (r * fn + fp) / n for r in C.COST_SWEEP},
        "cost_primary_R": C.COST_RATIO,
        "cost_primary": (C.COST_RATIO * fn + fp) / n,
    }
    if score is not None:
        out["AUROC"] = float(roc_auc_score(y, score))
    return out


def auroc_bootstrap_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Stratified (fails and passes resampled separately) percentile bootstrap 95% CI for AUROC. Added post-hoc."""
    rng = np.random.default_rng(seed)
    f, p = np.where(y == 1)[0], np.where(y == 0)[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(f, len(f)), rng.choice(p, len(p))])
        out[b] = roc_auc_score(y[idx], score[idx])
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def trivial_costs(y: np.ndarray) -> dict:
    n, nf = len(y), int(y.sum())
    return {
        "never_alarm": {str(r): r * nf / n for r in C.COST_SWEEP},
        "always_alarm": {str(r): (n - nf) / n for r in C.COST_SWEEP},
    }
