# Predictive Maintenance Pipeline — Context Document for Code Review

## Purpose
This document describes the full predictive maintenance pipeline. Use it as context
when reviewing the three attached code files for correctness, consistency, and
alignment with the intended pipeline.

---

## Goal
Given a CSV of sensor readings collected over the **entire lifecycle** of a machine
(from fault onset on day 1 through failure on the last day), the pipeline must:
1. Identify **which fault type** is present (fault classification).
2. Predict the **severity** of that fault for each day.
3. Given a user-chosen cutoff day (simulating "we are currently on day N"),
   estimate the **Remaining Useful Life (RUL)** — i.e., how many days remain
   before fault severity crosses the known failure threshold.
4. Plot the predicted degradation curve against the true full-lifecycle curve
   to validate the prediction.

---

## Files Overview

| File | Role |
|---|---|
| `model_training.py` | One-time training script. Trains and saves all models. |
| `fault_prediction.py` | Inference script. Loads saved models, runs on full-lifecycle input CSV, outputs `rul_input.csv`. |
| `rul_prediction.py` | RUL script. Reads `rul_input.csv`, fits a degradation curve using only N days of data, extrapolates to failure, plots predicted vs. true full lifecycle. |

---

## Input Data

### Training data: `dataset_latest.csv`
Used **only** in `model_training.py`. Contains labelled sensor data across many
machines/conditions. Columns:
- **14 sensor feature columns** (model inputs):
  `fPeak, pLow, pMid, pHigh, pKurtosis, qMean, qVar, qSkewness, qKurtosis,
  qPeak2Peak, qCrest, qRMS, qMAD, qCSRange`
- **3 severity columns** (regression targets):
  `LeakFault, BlockingFault, BearingFault`
- **`FaultClass`** (classification target): integer 0=Healthy, 1=Leak, 2=Blocking, 3=Bearing
- **`current_day`**: day index — not a model feature, used for identification only
- **`total_lifecycle`**: present in CSV but not used in this pipeline

### Inference input: `machine_lifecycle_input.csv`
Used in `fault_prediction.py`. This is a **full-lifecycle CSV for one machine** —
every row is one day, starting from day 1 (fault onset) through the final day
(when severity reaches the failure threshold). The severity values follow an
exponential-like degradation curve with added noise. This CSV has the same column
structure as the training data but is **not** the training CSV itself — it is new,
unseen data for a specific machine being monitored.

Must contain: all 14 feature columns, all 3 severity columns (kept for validation,
NOT fed to models), `FaultClass` (kept for validation, NOT fed to models), and
`current_day`.

### Intermediate file: `rul_input.csv`
Written by `fault_prediction.py`, read by `rul_prediction.py`.
Contains **all rows** from the full-lifecycle input (one row per day).
Columns: `current_day`, `severity`, `fault_class`.

---

## File 1: `model_training.py`

### What it does
- Loads `dataset_latest.csv`
- Separates features (`X`), classification target (`y_cls = FaultClass`), and
  regression target (`y_reg = [LeakFault, BlockingFault, BearingFault]`)
- Splits into train/test (80/20) **before** scaling to prevent data leakage
- Fits a `StandardScaler` on training features only; transforms both train and test
- Trains a `RandomForestClassifier` (with `class_weight='balanced'`) on `y_cls`
- Trains a `RandomForestRegressor` (multi-output) on all 3 severity columns at once
- Prints metrics: classifier accuracy + classification report; per-column MAE, RMSE,
  R² for the regressor
- Saves three `.pkl` files:
  - `fault_classifier.pkl`
  - `fault_regressor.pkl`
  - `feature_scaler.pkl`

### Critical constraints
- Train/test split must happen before scaling (no data leakage).
- The regressor predicts all 3 severity columns simultaneously (multi-output),
  NOT one column at a time and NOT one fault type only.
- Both models use the same scaler and the same 14-column feature set.
- All three `.pkl` files must be saved with exactly those names.

---

## File 2: `fault_prediction.py`

### What it does
1. Loads `fault_classifier.pkl`, `fault_regressor.pkl`, `feature_scaler.pkl`
2. Reads `machine_lifecycle_input.csv` — the **full lifecycle** of one machine
   (all days from fault start to end of life)
3. Extracts only the 14 feature columns → scales with the loaded scaler
4. **Classifier** predicts `FaultClass` for every row (day)
5. **Regressor** predicts all 3 severity values for every row
6. **Majority vote** across all rows → determines the single final `FaultClass`
   for this machine
7. Prints a **validation table**: for each row, shows `current_day`,
   predicted severity, and true severity (from CSV) — only for the severity
   column that corresponds to the final fault class
8. Saves `rul_input.csv` with **all rows** and three columns:
   - `current_day`: day index from the input CSV
   - `severity`: predicted severity for the winning fault class only
     (the other two severity columns are ignored — they remain at baseline
     for a machine with only one active fault)
   - `fault_class`: the final majority-voted fault class (same value every row)

### Critical constraints
- `current_day`, severity columns, and `FaultClass` are NOT fed to the models.
- Only the 14 feature columns go through the scaler and into the models.
- ALL rows (full lifecycle) are written to `rul_input.csv` — not a subset.
- The output CSV must contain exactly: `current_day`, `severity`, `fault_class`.

---

## File 3: `rul_prediction.py`

### What it does
1. Reads `rul_input.csv` — contains the **full lifecycle** (all days)
2. Sorts rows by `current_day`
3. User sets `HOW_MANY_DAYS_TO_USE` — only that many rows (from day 1) are
   used for curve fitting. This simulates "we are currently on day N and want
   to predict how many days remain." The rest of the CSV rows are used only
   for plotting the true curve as a validation reference.
4. Determines the failure threshold based on fault class:
   - Class 1 (Leak):     threshold = 0.036   (severity INCREASES toward threshold)
   - Class 2 (Blocking): threshold = 0.53    (severity DECREASES toward threshold)
   - Class 3 (Bearing):  threshold = 6e-4    (severity INCREASES toward threshold)
5. Smooths the N-day severity history with a 3-day moving average
6. Fits a **power-law degradation curve** to the smoothed history:
   - For increasing faults: `S(d) = S_start + (S_threshold - S_start) * ((d - d0) / (L - d0))^alpha`
   - For decreasing faults (blockage): `S(d) = S_start - (S_start - S_threshold) * ((d - d0) / (L - d0))^alpha`
   - `S_start` = severity value on day 1 of the CSV — taken from data, never hardcoded
   - `d0` = `days_hist[0] - 1` (so the curve anchors exactly at S_start on the first day)
   - `L` = predicted total lifecycle (days) — the parameter being solved for
   - `alpha` = degradation shape, bounded `[ALPHA_LB=1.5, ALPHA_UB=3.0]`
   - Solver bounds enforce `L > current_day` (failure must be in the future)
7. RUL = `L_pred - current_day`
8. Prints: predicted lifecycle, alpha, RUL
9. Produces `rul_curve_fitting_plot.png` with:
   - **Red scatter**: raw predicted severity for the N days used for fitting
   - **Orange dashed**: 3-day smoothed severity (what the curve was actually fit to)
   - **Solid blue**: fitted curve over the N-day history period
   - **Dashed blue**: extrapolated curve from current day onward to predicted failure
   - **Green dash-dot**: true severity for ALL days in the CSV (full lifecycle reference)
   - **Magenta vertical**: current day marker
   - **Red horizontal**: failure threshold line
   - X-axis extends to cover both the predicted lifecycle end AND the true last day
     in the CSV, whichever is larger — so the green true curve is never cut off

### Critical constraints
- The blockage fault (class 2) uses an INVERTED curve: starts high, decays toward
  the threshold from above. All other faults start low and increase.
- `S_start` is always the first row's severity value — derived from data, not hardcoded.
- Only the first `HOW_MANY_DAYS_TO_USE` rows go into the curve fitter.
- The full CSV (all rows) is plotted as the true reference curve.
- `x_max` for the plot must use `df['current_day'].max()` (the actual last day value),
  NOT `len(df)` (row count) — these differ if `current_day` doesn't start at 1
  or has gaps.

---

## Data Flow Summary

```
dataset_latest.csv
       |
       v
model_training.py
       |
       |---> fault_classifier.pkl
       |---> fault_regressor.pkl
       +---> feature_scaler.pkl
                    |
                    v
machine_lifecycle_input.csv ---> fault_prediction.py
(full lifecycle, all days)              |
                                        |  ALL rows written
                                        v
                                   rul_input.csv
                               (current_day, severity, fault_class)
                               (full lifecycle, all days)
                                        |
                                        v
                               rul_prediction.py
                               (user sets HOW_MANY_DAYS_TO_USE
                                to simulate current day N)
                                        |
                                        v
                        RUL value + plot: predicted curve vs.
                        true full-lifecycle curve
```

---

## What to Check When Reviewing the Code

1. **`model_training.py`**: Is the train/test split done before scaling? Is the
   regressor trained on all 3 severity columns at once (multi-output, not just one)?
   Are all 3 `.pkl` files saved with the correct names?

2. **`fault_prediction.py`**: Are only the 14 feature columns passed to the scaler
   and models? Is majority voting used for the final fault class? Are ALL rows
   (full lifecycle) written to `rul_input.csv`? Does the output CSV contain exactly
   `current_day`, `severity` (winning fault class only), and `fault_class`?

3. **`rul_prediction.py`**: Is the curve direction correct per fault class
   (blockage decreases, others increase)? Is `S_start` taken from the data?
   Is fitting restricted to the first `HOW_MANY_DAYS_TO_USE` rows while the full
   CSV is plotted as the green reference? Does `x_max` use `df['current_day'].max()`
   not `len(df)`?

4. **Cross-file consistency**: Do the column names in `rul_input.csv` written by
   `fault_prediction.py` exactly match what `rul_prediction.py` reads?
   Do the fault class integers (1, 2, 3) and their threshold values agree
   across both files?
