Test block: 627 runs (28 fails, 599 passes). Thresholds at α = 1% from Phase I calibration passes (I-MR rows are not calibrated). Cost at R = 10 (assumed); lower is better. Reference costs: never alarm = 0.4466, always alarm = 0.9553.

| method | detected fails | detection rate (95% CI) | false-alarm rate | ARL0 emp. (1/FAR) | ARL1 / median delay / ≤5 runs / censored | AUROC | cost @R=10 |
|---|---|---|---|---|---|---|---|
| PCA Hotelling T² (classic) | 0/28 | 0.000 (0.00–0.12) | 0.022 (13/599) | 37.8 (46.1) | 44.9 / 48 / 0.29 / 1 | 0.424 | 0.4673 |
| PCA SPE/Q (classic) | 0/28 | 0.000 (0.00–0.12) | 0.042 (25/599) | 20.1 (24.0) | 39.5 / 38 / 0.25 / 1 | 0.474 | 0.4864 |
| Isolation Forest (ML, unsupervised) | 2/28 | 0.071 (0.02–0.23) | 0.030 (18/599) | 31.8 (33.3) | 25.4 / 19 / 0.21 / 0 | 0.554 | 0.4434 |
| Autoencoder MLP (ML, unsupervised) | 0/28 | 0.000 (0.00–0.12) | 0.033 (20/599) | 25.1 (30.0) | 43.1 / 46 / 0.25 / 1 | 0.441 | 0.4785 |
| Gradient boosting (ML, SUPERVISED upper-bound ref, uses labels) | 1/28 | 0.036 (0.01–0.18) | 0.018 (11/599) | 54.3 (54.5) | 22.6 / 12 / 0.21 / 0 | 0.598 | 0.4482 |
| I-MR, WE rules 1-4 + MR, top-10 sensors (classic, uncalibrated) | 28/28 | 1.000 (0.88–1.00) | 1.000 (599/599) | 1.0 (1.0) | 1.0 / 0 / 1.00 / 0 | n/a | 0.9553 |
| I-MR, rule 1 only, top-10 sensors (classic, uncalibrated) | 12/28 | 0.429 (0.27–0.61) | 0.369 (221/599) | 2.7 (2.7) | 2.8 / 1 / 0.93 / 0 | n/a | 0.6077 |

Cost sensitivity (cost per test run; thresholds fixed at α = 1%):

| method | R=1 | R=5 | R=10 | R=20 | R=50 |
|---|---|---|---|---|---|
| PCA_T2 | 0.0654 | 0.2440 | 0.4673 | 0.9139 | 2.2536 |
| PCA_SPE | 0.0845 | 0.2632 | 0.4864 | 0.9330 | 2.2727 |
| IsolationForest | 0.0702 | 0.2360 | 0.4434 | 0.8581 | 2.1021 |
| Autoencoder | 0.0766 | 0.2552 | 0.4785 | 0.9250 | 2.2648 |
| GradBoost_SUPERVISED | 0.0606 | 0.2329 | 0.4482 | 0.8788 | 2.1707 |
| IMR_WE_rules_top10 | 0.9553 | 0.9553 | 0.9553 | 0.9553 | 0.9553 |
| IMR_rule1_only_top10 | 0.3780 | 0.4801 | 0.6077 | 0.8628 | 1.6284 |
| *never_alarm* | 0.0447 | 0.2233 | 0.4466 | 0.8931 | 2.2329 |
| *always_alarm* | 0.9553 | 0.9553 | 0.9553 | 0.9553 | 0.9553 |

Secondary (not primary; same test block):

| variant | detected | FAR | cost @R=10 |
|---|---|---|---|
| PCA_T2 @alpha=0.27% | 0/28 | 0.012 | 0.4577 |
| PCA_SPE @alpha=0.27% | 0/28 | 0.010 | 0.4561 |
| IsolationForest @alpha=0.27% | 1/28 | 0.020 | 0.4498 |
| Autoencoder @alpha=0.27% | 0/28 | 0.013 | 0.4593 |
| GradBoost_SUPERVISED @alpha=0.27% | 1/28 | 0.005 | 0.4354 |
| PCA_T2 @parametric F 99% | 25/28 | 0.925 | 0.9314 |
| PCA_SPE @parametric Jackson-Mudholkar 99% | 28/28 | 1.000 | 0.9553 |
| PCA_T2_or_SPE @alpha=1% each | 0/28 | 0.048 | 0.4928 |
