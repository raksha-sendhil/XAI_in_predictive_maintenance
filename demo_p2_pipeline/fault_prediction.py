"""
fault_prediction.py
--------------------
Pipeline:
  1. Load trained classifier, regressor, and scaler from .pkl files.
  2. Read input CSV (sensor features + fault-class + 3 severity cols + current_day).
  3. Run classifier  → predicted FaultClass per row.
  4. Run regressor   → predicted severity for all 3 fault types per row.
  5. Determine final (majority-vote) fault class.
  6. Print validation table: day | predicted severity | true severity.
  7. Save output CSV: current_day | severity | true_severity | fault_class
     (severity      = predicted severity for the winning fault class column)
     (true_severity = ground-truth severity for the winning fault class column)

Output file : rul_input.csv
"""

import numpy as np
import pandas as pd
import joblib
from collections import Counter

# =============================================================================
# CONFIGURATION  ← edit paths here
# =============================================================================

INPUT_CSV      = "machine_lifecycle_input.csv"
OUTPUT_CSV     = "rul_input.csv"

CLASSIFIER_PKL = "fault_classifier.pkl"
REGRESSOR_PKL  = "fault_regressor.pkl"
SCALER_PKL     = "feature_scaler.pkl"

FEATURE_COLS = [
    'fPeak', 'pLow', 'pMid', 'pHigh', 'pKurtosis',
    'qMean', 'qVar', 'qSkewness', 'qKurtosis',
    'qPeak2Peak', 'qCrest', 'qRMS', 'qMAD', 'qCSRange'
]

SEVERITY_COLS = ['LeakFault', 'BlockingFault', 'BearingFault']

FAULT_CLASS_MAP = {
    0: None,
    1: 'LeakFault',
    2: 'BlockingFault',
    3: 'BearingFault',
}

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
classifier = joblib.load(CLASSIFIER_PKL)
regressor  = joblib.load(REGRESSOR_PKL)
scaler     = joblib.load(SCALER_PKL)
print("  classifier :", CLASSIFIER_PKL)
print("  regressor  :", REGRESSOR_PKL)
print("  scaler     :", SCALER_PKL)

# =============================================================================
# LOAD INPUT DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded '{INPUT_CSV}': {len(df)} rows, {len(df.columns)} columns")

required = FEATURE_COLS + SEVERITY_COLS + ['FaultClass', 'current_day']
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV is missing columns: {missing}")

X = df[FEATURE_COLS]

# =============================================================================
# SCALE FEATURES
# =============================================================================

X_scaled = scaler.transform(X)

# =============================================================================
# PREDICT FAULT CLASS (classifier)
# =============================================================================

predicted_classes     = classifier.predict(X_scaled)
df['pred_FaultClass'] = predicted_classes

print("\n--- Predicted FaultClass per row ---")
for i, row in df.iterrows():
    print(f"  Day {int(row['current_day']):>3} | "
          f"Predicted: {int(row['pred_FaultClass'])}  "
          f"True: {int(row['FaultClass'])}")

# =============================================================================
# MAJORITY-VOTE FINAL FAULT CLASS
# =============================================================================

counts       = Counter(predicted_classes)
final_class  = counts.most_common(1)[0][0]
severity_col = FAULT_CLASS_MAP.get(final_class)

print(f"\n--- Fault Class Vote Summary ---")
for cls, cnt in sorted(counts.items()):
    label = FAULT_CLASS_MAP.get(cls) or "Healthy"
    print(f"  Class {cls} ({label}): {cnt} rows")

print(f"\n>>> FINAL PREDICTED FAULT CLASS : {final_class} "
      f"({FAULT_CLASS_MAP.get(final_class) or 'Healthy'})")

if severity_col is None:
    print("Machine is predicted HEALTHY — no RUL calculation needed.")

# =============================================================================
# PREDICT SEVERITY (regressor) for all 3 fault types
# =============================================================================

predicted_severity = regressor.predict(X_scaled)   # shape: (n_rows, 3)

for idx, col in enumerate(SEVERITY_COLS):
    df[f'pred_{col}'] = predicted_severity[:, idx]

# =============================================================================
# VALIDATION TABLE
# =============================================================================

if severity_col is not None:
    pred_col_name = f'pred_{severity_col}'
    print(f"\n--- Validation Table (fault: {severity_col}) ---")
    header = f"{'Day':>5} | {'Pred Severity':>18} | {'True Severity':>18}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        print(f"  {int(row['current_day']):>3}  | "
              f"{row[pred_col_name]:>18.6e} | "
              f"{row[severity_col]:>18.6e}")
else:
    print("\n(No validation table — machine predicted healthy.)")

# =============================================================================
# SAVE OUTPUT CSV  →  rul_input.csv
# Includes both predicted severity (reference) and true_severity (for RUL fitting)
# =============================================================================

if severity_col is not None:
    out_df = pd.DataFrame({
        'current_day'   : df['current_day'].values,
        'severity'      : df[f'pred_{severity_col}'].values,   # predicted (kept for reference)
        'true_severity' : df[severity_col].values,             # ground-truth  ← NEW
        'fault_class'   : final_class,
    })

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved RUL input CSV to '{OUTPUT_CSV}' ({len(out_df)} rows).")
    print(out_df.head())
else:
    print("\nNo output CSV saved (healthy machine).")
