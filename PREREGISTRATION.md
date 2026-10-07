# Preregistration — secom-spc-vs-ai v0

Written and committed (git tag `prereg`) **before any detection method was run** on the data.
Everything below is fixed in advance. Any deviation discovered later will be listed in the README
under "Deviations from preregistration" rather than silently changed here.

What had already been looked at before writing this: the file shapes (1,567 rows x 590 sensors),
the label counts (104 fail / 1,463 pass), the timestamp range and order, the number of fails that fall
in each split below, and the overall missing fraction (~4.5% of cells). No model, chart, or score had been computed on the test block (runs 940–1566). The analysis code committed with this file
(`secom/`) was smoke-tested only on the training period (runs 0–939, calibration block standing in as a pseudo-test) to check that it runs; those
smoke numbers are in-sample, were not used to change any choice below, and are not reported.

## Data

- Source: UCI Machine Learning Repository, SECOM (dataset 179),
  `https://archive.ics.uci.edu/static/public/179/secom.zip` (files `secom.data`, `secom_labels.data`).
  SHA-256 checksums go in `data/MANIFEST.json`; raw data is gitignored.
- Real data only. Nothing synthetic, no oversampling, no SMOTE.
- Label: `-1` = pass, `1` = fail. "Fail" is the positive class (the thing we want to alarm on).
- Sensor columns are anonymized in the source. They are named `sensor_000` … `sensor_589` after their
  0-based column position in `secom.data`. We don't know or guess what any sensor measures.

## Time order

- Runs are sorted by timestamp (`dd/mm/yyyy HH:MM:SS`) using a **stable** sort, so runs that share a
  timestamp keep their original file order. (The file is already in non-decreasing timestamp order; 33 rows share a timestamp
  with the row before them.) "Run order" below means this sorted position, 0 … 1566.

## Split (time-ordered, exact cutoffs)

| block | run positions | n runs | time range | fails | passes | used for |
|---|---|---|---|---|---|---|
| Phase I – fit | 0 – 625 | 626 | 2008-07-19 11:55 → 2008-09-01 06:21 | 65 | 561 | fit preprocessing + models |
| Phase I – calibration | 626 – 939 | 314 | 2008-09-01 06:52 → 2008-09-20 05:34 | 11 | 303 | set alarm thresholds / control limits |
| Phase II – test | 940 – 1566 | 627 | 2008-09-20 06:08 → 2008-10-17 06:07 | 28 | 599 | final evaluation only |

The cutoffs come from proportions fixed in advance: training period = first 60% of runs (floor(0.6·1567) = 940);
inside that, Phase I-fit = first two-thirds (floor(940·2/3) = 626) and calibration = the remaining 314 runs; test = the last 40% (627 runs).

- **Phase I baseline = PASSING runs only.** Unsupervised methods (PCA/T²/SPE, I-MR, Isolation Forest,
  autoencoder) are fit on the 561 passing runs of Phase I-fit. Control limits/thresholds come from the 303
  passing runs of Phase I-calibration (held out from the fit, so the limits aren't tuned on the fit data).
- The supervised reference (gradient boosting) is the only method that sees labels. It is fit on all 626 Phase I-fit runs
  (fails included), and its threshold is set the same way, on Phase I-calibration passes.
- The test block is used once, for the final numbers.

## Preprocessing (statistics from the training period only, never the test block)

1. Missing-value audit: per-sensor missing % and constant sensors are reported for the full dataset (descriptive only) and
   for the training period (used for decisions).
2. Drop sensors with > 50% missing values in the training period (runs 0–939; feature values only, labels not used).
3. Median-impute remaining missing values with medians from the Phase I-fit passing runs.
4. Drop sensors that are constant (std = 0) on the Phase I-fit passing runs after imputation.
5. Standardize (z-score) with mean/std from the Phase I-fit passing runs.
6. The same fitted transform is applied unchanged to calibration and test runs.

## Methods

Classic SPC
- **PCA + Hotelling T²** and **PCA + SPE/Q**. PCA is fit on standardized Phase I-fit passes. We keep the smallest number of components
  that explains ≥ 90% of the variance. T² = Σ t_a²/λ_a over retained components; SPE = ‖x − x̂‖².
- **Univariate individuals (I-MR) charts** on the top 10 sensors. Sensors are ranked on Phase I-fit runs only, by |AUROC − 0.5| of the
  (imputed) sensor value vs. the fail label. Center line and σ (= MR̄/1.128) come from the Phase I-fit passing runs in run order.
  Signals: Western Electric rules — (1) 1 point beyond 3σ; (2) 2 of 3 consecutive beyond 2σ on the same side;
  (3) 4 of 5 consecutive beyond 1σ on the same side; (4) 8 consecutive on the same side of the center line; plus the MR chart
  (MR > 3.267·MR̄). A run alarms if any rule fires on any of the 10 sensors at that run. Rules are evaluated over the full series in run order
  (monitoring continues from Phase I into test), and only test runs are scored. Two variants get reported:
  `I-MR (WE rules 1-4 + MR)`, the shop-floor default (NOT calibrated to the target false-alarm rate), and
  `I-MR rule 1 only`, which is also not calibrated. Both are reported exactly as they come out, so the multiple-testing false-alarm problem stays visible.

Machine learning
- **Isolation Forest**, unsupervised: fit on Phase I-fit passes, 500 trees, `random_state=0`. Score = −`score_samples`.
- **Autoencoder** (secondary, CPU): sklearn `MLPRegressor` with hidden layers (64, 16, 64), ReLU, fit to reconstruct standardized
  Phase I-fit passes, `random_state=0`, max 500 iters. Score = reconstruction MSE.
- **Gradient boosting: SUPERVISED UPPER-BOUND REFERENCE, uses labels, never a competitor**: sklearn `HistGradientBoostingClassifier` (`class_weight="balanced"`,
  `random_state=0`, defaults otherwise) trained on Phase I-fit runs **with labels**. It is labeled as using labels everywhere. In practice an engineer often
  doesn't have labels this early (they're only known after test/sort), so this row is a ceiling to compare against, not a fair competitor.

## Control limits / thresholds

- **Primary target false-alarm rate α = 1% (99% limit)**. For every scored method (T², SPE, Isolation Forest, autoencoder, gradient boosting),
  the alarm threshold is the empirical 99th percentile of that method's score on the 303 Phase I-calibration passing runs.
  Alarm = score > threshold.
- Secondary (reported, not primary): α = 0.27% (99.73%, the 3σ convention) set the same way; and for T²/SPE, the textbook parametric limits
  (T²: F-distribution limit at 99%; SPE: Jackson–Mudholkar at 99%) computed from Phase I-fit, to show how far they are from the empirical limits.

## Metrics (all on the 627 test runs, in run order)

- **Detection rate (DR)** = alarmed test fails / 28.
- **False-alarm rate (FAR)** = alarmed test passes / 599.
- **ARL0 (in-control average run length)**: (a) 1/FAR (geometric approximation); (b) empirical: mean number of passing runs between
  consecutive false alarms in the test pass sequence, counting from the first test pass (gaps after the last alarm are not counted). Reported as ∞ if there are no false alarms.
- **Detection delay / ARL1**: for each test fail at run position i, delay_i = j − i, where j ≥ i is the first test run position (any label)
  that alarms. delay = 0 means the fail run itself alarmed. If no alarm happens before the end of the test block, the fail is censored and
  delay_i = (1566 − i) is recorded as a lower bound. Reported: ARL1 = mean(delay_i + 1), median delay, fraction detected within 5 runs (delay ≤ 5),
  and the number censored. Caveat stated in advance: at FAR around 1%, a random alarm will arrive within about 100 runs anyway, so delay is only
  meaningful next to ARL0.
- **AUROC** of the continuous score (threshold-free, secondary).
- **Cost-weighted score (ASSUMPTION)**: missed fail (escape) costs R, false alarm costs 1, true detections and correct passes cost 0.
  Cost per test run = (R·FN + FP) / 627; lower is better. **Primary ratio R = 10** is an assumption (an escape is taken to be 10× as costly as investigating a
  false alarm); it isn't a measured fab cost. Each method is compared to the trivial "never alarm" policy (cost = R·28/627) and the "always alarm" policy
  (cost = 599/627).
- **Sensitivity sweep**: R ∈ {1, 5, 10, 20, 50}, thresholds fixed at the primary α = 1% (they are not re-tuned per R).

## Contribution analysis

For each test alarm from T² and SPE: per-sensor contributions. T²: c_j = x_j · (P Λ⁻¹ Pᵀ x)_j, which sums to T². SPE: c_j = e_j², which sums to SPE.
Top-5 contributing sensors per alarm are saved to CSV. Bar plots cover the top alarms, along with how often each sensor shows up among the top-5 contributors across all
test alarms. Sensors are anonymized, so the output says where to look (which column), not what it physically is.

## Decision rule (stated before running)

ML "beats" classic SPC for this dataset only if, at the same calibrated α = 1%, an unsupervised ML method (Isolation Forest or autoencoder)
has a higher test detection rate **and** a lower cost at R = 10 than the better of T² and SPE. With only 28 test fails, a difference of 1–2 detections
is within noise (each fail = 3.6 percentage points of DR). A 95% Wilson interval on DR will be reported, and no difference will be called decisive if the intervals overlap heavily.

## Likely honest outcome (stated before running)

SECOM is a known hard dataset: fails are rare, the sensors are noisy and anonymized, and many fails probably don't show up in these signals at all.
Our prediction, written down before any test-block number exists:
- At a calibrated 1% false-alarm budget, **every unsupervised method (classic or ML) will catch only a small share of test fails**, likely well under
  a third, and **control charts (T²/SPE) will tie or beat the unsupervised ML methods** within the noise of 28 test fails.
- Process drift after Phase I will push test false-alarm rates **above** the nominal 1% for all static Phase I models.
- Univariate I-MR charts with all Western Electric rules on 10 sensors will drown in false alarms (multiple testing, no calibration).
- The supervised gradient-boosting **upper-bound reference (it uses labels)** may beat the unsupervised methods on AUROC, but it isn't a competitor:
  it answers "how much signal is there if you already have labeled fails," not "which monitor should run on the line."
- It's quite possible that **no method beats "never alarm" on cost at R = 10**. If that's what happens, it is the finding and gets reported as such.
None of these predictions changes any analysis choice above. The bar does not move after the test block is scored.

## Known weakness of this preregistration

The repository is private, and the author controls git dates and tags, so this `prereg` tag alone doesn't prove the plan came before the results to an outside skeptic.
The fix (left for later, needs owner approval): publish the `prereg` commit SHA somewhere outside our control before or at the time the repo goes public.

## Reproducibility

`make all` downloads the data, verifies checksums, runs every method with fixed seeds, writes `results/results.json`,
regenerates the figures in `figures/`, and regenerates the README results table from the JSON.
