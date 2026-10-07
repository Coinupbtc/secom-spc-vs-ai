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

1. **Machine for the official run.** The plan commit was authored on a second machine. A first `make all` attempt there after the tag was an
   interrupted run, and no metrics were produced (only the descriptive `missing_by_sensor.csv` was written, then deleted). The official run, whose numbers are committed,
   was done once on the primary machine (CPU only) from the tagged code. `data/MANIFEST.json` was written at the first download, and the later download matched all four SHA-256 values.
2. No analysis choice in the planned analysis was changed after the test block was scored. The README interpretation text was written after the results;
   the tables are generated from `results/results.json`.
3. **History rewritten to remove a sensitive guard token; analysis files byte-identical to b3f2a36.** The original guard script stored a sensitive token.
   Both original commits were rebuilt with a generic guard (`scripts/identity_guard.py`, `.github/workflows/identity-guard.yml`, which contain no guarded values).
   Every other file is byte-identical to the originals, and commit messages and author dates are unchanged. The plan tag was re-cut on the rebuilt plan commit.
   Check: run the following on the plan commit (the commit the `prereg` tag points to).
   ```bash
   GIT_INDEX_FILE=/tmp/idx git read-tree <plan-commit>
   GIT_INDEX_FILE=/tmp/idx git rm -q -f --cached scripts/identity_guard.py .github/workflows/identity-guard.yml
   GIT_INDEX_FILE=/tmp/idx git write-tree      # -> 24b35ac6dd6bf8d0cf3221eb555facb10035f9d6
   git ls-tree -r <plan-commit> | grep -v -P '\t(scripts/identity_guard\.py|\.github/workflows/identity-guard\.yml)$' | sha256sum
                                                # -> 9940bc64f9da6867979c7cf6fc9e15b1644287821c2088025099e76120555dc4
   ```
   Both values are identical for the original plan commit `b3f2a36` and the rebuilt one.
4. **Wording changes after review (no effect on numbers):** the gradient-boosting row is now called "supervised reference (uses labels)". The frozen plan
   says "upper-bound reference", but one model is not a ceiling. The README says "analysis plan frozen before the test block" instead of "preregistered".
   AUROCs whose CI includes 0.5 are described as indistinguishable from chance. The earlier README wording that attributed the too-tight textbook limits to drift alone was withdrawn (see item 5).
   `results.json` now carries a UTC timestamp.
5. **Exploratory, post-hoc analyses (added after the test block was scored; NOT part of the frozen plan).** The code is in `secom/exploratory.py` and `secom/run.py`, and the outputs are under
   `exploratory_posthoc` in `results/results.json`. The planned-analysis numbers are unchanged; this was checked field by field against the previous `results.json`.
   - K2: random (non-time) 25% holdout of Phase I-fit passes, 20 seeds, empirical-vs-textbook T²/SPE limit ratios on the random holdout vs. later blocks.
     The reading rule was written in the module docstring before running.
   - K3: share of Western Electric rule-4 hits whose 8-point window contains a median-imputed point, plus an I-MR variant where rule 4 skips imputed points.
   - Stratified bootstrap 95% CIs (2,000 resamples) on test AUROC.
   - Random-split AUROC: the same pipeline code on a randomly permuted 626/314/627 split (10 seeds), compared with the time-ordered split.
