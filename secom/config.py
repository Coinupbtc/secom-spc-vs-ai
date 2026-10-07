"""Preregistered constants (see PREREGISTRATION.md). Do not change without logging a deviation."""
N_TOTAL = 1567
FIT = (0, 626)          # Phase I - fit       (runs 0..625)
CAL = (626, 940)        # Phase I - calibration (runs 626..939)
TEST = (940, 1567)      # Phase II - test     (runs 940..1566)
TRAIN = (0, 940)        # training period for missingness filter
MAX_MISSING = 0.50
PCA_VAR = 0.90
ALPHA = 0.01            # primary target false-alarm rate
ALPHA_SECONDARY = 0.0027
COST_RATIO = 10         # ASSUMPTION: escape cost / false-alarm cost
COST_SWEEP = [1, 5, 10, 20, 50]
IMR_TOP_K = 10
SEED = 0
