"""
model_training.py
------------------
Trains and saves:
  - fault_classifier.pkl  : RandomForestClassifier  → predicts FaultClass (0/1/2/3)
  - fault_regressor.pkl   : RandomForestRegressor   → predicts all 3 severity columns
                            (LeakFault, BlockingFault, BearingFault) simultaneously
  - feature_scaler.pkl    : StandardScaler fitted on training features only
  - model_metrics.json    : Classifier + regressor metrics for the Streamlit UI

Run this ONCE. After the .pkl files are saved, use fault_prediction.py for inference.
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_CSV = "dataset_latest.csv"

FEATURE_COLS = [
    'fPeak', 'pLow', 'pMid', 'pHigh', 'pKurtosis',
    'qMean', 'qVar', 'qSkewness', 'qKurtosis',
    'qPeak2Peak', 'qCrest', 'qRMS', 'qMAD', 'qCSRange'
]

# All three severity columns — regressor predicts all of them at once
SEVERITY_COLS = ['LeakFault', 'BlockingFault', 'BearingFault']

FAULT_CLASS_COL = 'FaultClass'

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(TRAINING_CSV)
print(f"Loaded '{TRAINING_CSV}': {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFaultClass distribution:")
print(df[FAULT_CLASS_COL].value_counts().sort_index())

X      = df[FEATURE_COLS]
y_cls  = df[FAULT_CLASS_COL]           # classification target
y_reg  = df[SEVERITY_COLS]             # regression target (multi-output)

# =============================================================================
# TRAIN / TEST SPLIT  (done BEFORE scaling to prevent data leakage)
# =============================================================================

X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls,
    test_size=TEST_SIZE,
    stratify=y_cls,
    random_state=RANDOM_STATE
)

# Use the same row indices for the regression split
y_train_reg = y_reg.loc[X_train.index]
y_test_reg  = y_reg.loc[X_test.index]

print(f"\nTrain size : {len(X_train)} | Test size : {len(X_test)}")

# =============================================================================
# SCALE FEATURES  (fit on train only, transform both)
# =============================================================================

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform on train
X_test_sc  = scaler.transform(X_test)        # transform only on test

# =============================================================================
# TRAIN CLASSIFIER
# =============================================================================

print("\nTraining RandomForestClassifier...")
classifier = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=RANDOM_STATE
)
classifier.fit(X_train_sc, y_train_cls)

y_pred_cls = classifier.predict(X_test_sc)
cls_acc    = accuracy_score(y_test_cls, y_pred_cls)

print(f"Classifier Accuracy : {cls_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls,
      target_names=['Healthy', 'LeakFault', 'BlockingFault', 'BearingFault']))

# =============================================================================
# TRAIN REGRESSOR  (multi-output — predicts all 3 severity columns at once)
# =============================================================================

print("Training RandomForestRegressor (multi-output)...")
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE
)
regressor.fit(X_train_sc, y_train_reg)

y_pred_reg = regressor.predict(X_test_sc)   # shape: (n_test, 3)

print("\nRegressor Metrics (per severity column):")
reg_metrics_dict = {}
for i, col in enumerate(SEVERITY_COLS):
    mae  = mean_absolute_error(y_test_reg.iloc[:, i], y_pred_reg[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_reg.iloc[:, i], y_pred_reg[:, i]))
    r2   = r2_score(y_test_reg.iloc[:, i], y_pred_reg[:, i])
    print(f"  {col:<20} MAE={mae:.4e}  RMSE={rmse:.4e}  R²={r2:.4f}")
    reg_metrics_dict[col] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# -- NEW: Calculate Overall Regressor Metrics --
overall_mae = mean_absolute_error(y_test_reg, y_pred_reg)
overall_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
overall_r2 = r2_score(y_test_reg, y_pred_reg)
print(f"\nOverall Regressor R²: {overall_r2:.4f}")

# =============================================================================
# SAVE MODELS
# =============================================================================

joblib.dump(classifier, 'fault_classifier.pkl')
joblib.dump(regressor,  'fault_regressor.pkl')
joblib.dump(scaler,     'feature_scaler.pkl')

print("\nSaved:")
print("  fault_classifier.pkl")
print("  fault_regressor.pkl")
print("  feature_scaler.pkl")

# =============================================================================
# SAVE MODEL METRICS
# =============================================================================

target_names = ['Healthy', 'LeakFault', 'BlockingFault', 'BearingFault']

cls_report_dict = classification_report(
    y_test_cls, y_pred_cls,
    target_names=target_names,
    output_dict=True
)

per_class_report = {name: cls_report_dict[name] for name in target_names}

metrics_payload = {
    "classifier": {
        "accuracy"               : float(cls_acc),
        "classification_report"  : per_class_report,
    },
    "regressor": {
        **reg_metrics_dict, # Unpacks the per-column dicts
        "overall_r2"   : float(overall_r2),
        "overall_mae"  : float(overall_mae),
        "overall_rmse" : float(overall_rmse)
    },
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics_payload, f, indent=2)

print("  model_metrics.json")
print("\nTraining complete. Run fault_prediction.py for inference.")