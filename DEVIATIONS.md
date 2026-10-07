# Deviations and pre-prereg exploration log

Everything done with the data before the `prereg` tag, and every departure from `PREREGISTRATION.md` after it, is listed here.

## Before the `prereg` tag (2026-10-07, ~4:22–4:35 PM CT)

1. **Data inspection (descriptive, no model):** file shapes (1567×590), label counts (104 fail / 1463 pass), timestamp range and order
   (already sorted; 33 rows share a timestamp with the previous row), fails per planned split (65 / 11 / 28), overall missing fraction (~4.5% of cells).
   The split proportions (60% train / 40% test, train split 2/3 fit + 1/3 calibration) were picked before the per-split fail counts were looked at.
   The counts were checked afterward and the proportions weren't changed.
2. **Code smoke test on the training period only.** To check that the pipeline runs, `secom/run.py` + `secom/report.py` were executed once on runs 0–939 only,
   with the Phase I-calibration block (runs 626–939) standing in as a pseudo-test. **No test-block run (940–1566) was loaded into any model or metric.**
   The smoke numbers are in-sample, because thresholds were set on the same calibration passes they were scored on, so they are meaningless as results. They are logged here for transparency only:

   | method (smoke, pseudo-test = calibration block, 11 fails / 303 passes) | DR | FAR | cost@R=10 |
   |---|---|---|---|
   | PCA_T2 | 0.000 | 0.013 | 0.3631 |
   | PCA_SPE | 0.000 | 0.013 | 0.3631 |
   | IsolationForest | 0.091 | 0.013 | 0.3312 |
   | Autoencoder | 0.000 | 0.013 | 0.3631 |
   | GradBoost_SUPERVISED (uses labels) | 0.273 | 0.013 | 0.2675 |
   | IMR_WE_rules_top10 | 1.000 | 1.000 | 0.9650 |
   | IMR_rule1_only_top10 | 0.364 | 0.360 | 0.5701 |

   Seeing these numbers changed no preregistered choice: methods, α, k, rules, top-k, and cost ratio are exactly as first drafted.
   Edits after the smoke test (no effect on analysis): removed the hostname field from `results.json` (identity hygiene), and added the "Likely honest outcome" and
   "Known weakness" sections to the preregistration.

## After the `prereg` tag

(none yet; anything added below this line is a deviation)
