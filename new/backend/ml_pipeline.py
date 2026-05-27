import numpy as np
import pandas as pd
import joblib
from collections import Counter
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d



# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONSTANTS  (mirroring the three .py files exactly)
# ─────────────────────────────────────────────────────────────────────────────


FEATURE_COLS = [
    'fPeak', 'pLow', 'pMid', 'pHigh', 'pKurtosis',
    'qMean', 'qVar', 'qSkewness', 'qKurtosis',
    'qPeak2Peak', 'qCrest', 'qRMS', 'qMAD', 'qCSRange'
]
SEVERITY_COLS   = ['LeakFault', 'BlockingFault', 'BearingFault']
FAULT_CLASS_MAP = {0: None, 1: 'LeakFault', 2: 'BlockingFault', 3: 'BearingFault'}
FAULT_NAMES     = {0: 'Healthy', 1: 'LeakFault', 2: 'BlockingFault', 3: 'BearingFault'}
FAULT_CONFIG    = {
    1: dict(name='LeakFault',     decreasing=False),
    2: dict(name='BlockingFault', decreasing=True),
    3: dict(name='BearingFault',  decreasing=False),
}
ALPHA_LB = 1.0
ALPHA_UB = 4.0
L_MAX    = 500

PKL_CLASSIFIER = "fault_classifier.pkl"
PKL_REGRESSOR  = "fault_regressor.pkl"
PKL_SCALER     = "feature_scaler.pkl"
METRICS_JSON   = "model_metrics.json"

FAULT_COLORS = {0: "#4caf50", 1: "#ef5350", 2: "#ffa726", 3: "#42a5f5"}
FAULT_CSS    = {0: "fault-0",  1: "fault-1",  2: "fault-2",  3: "fault-3"}



# ─────────────────────────────────────────────────────────────────────────────
# HELPER — fault_prediction logic (inlined)
# ─────────────────────────────────────────────────────────────────────────────


def run_fault_prediction(df_input, classifier, regressor, scaler):
    """
    Replicates fault_prediction.py logic.
    Returns: (rul_df, final_class, per_row_predictions_df, error_str)
    """
    missing = [c for c in FEATURE_COLS + SEVERITY_COLS + ['FaultClass', 'current_day']
               if c not in df_input.columns]
    if missing:
        return None, None, None, f"CSV missing columns: {missing}"

    X        = df_input[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # Classifier
    predicted_classes          = classifier.predict(X_scaled)
    df_input                   = df_input.copy()
    df_input['pred_FaultClass'] = predicted_classes

    # Majority vote
    counts      = Counter(predicted_classes)
    final_class = counts.most_common(1)[0][0]
    severity_col = FAULT_CLASS_MAP.get(final_class)

    # Regressor
    predicted_severity = regressor.predict(X_scaled)
    for idx, col in enumerate(SEVERITY_COLS):
        df_input[f'pred_{col}'] = predicted_severity[:, idx]

    if severity_col is None:
        return None, 0, df_input, None   # healthy

    # Build rul_input style dataframe
    rul_df = pd.DataFrame({
        'current_day'   : df_input['current_day'].values,
        'severity'      : df_input[f'pred_{severity_col}'].values,
        'true_severity' : df_input[severity_col].values,
        'fault_class'   : final_class,
    })

    return rul_df, final_class, df_input, None

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — rul_prediction logic (inlined)
# ─────────────────────────────────────────────────────────────────────────────
def run_rul_prediction(rul_df, how_many_days):
    """
    Replicates rul_prediction.py logic using PREDICTED severity.
    Returns dict with all results, or raises on failure.
    """
    df = rul_df.sort_values('current_day').reset_index(drop=True)

    fault_class = int(df['fault_class'].iloc[0])
    cfg         = FAULT_CONFIG[fault_class]
    decreasing  = cfg['decreasing']

    # NEW: Use predicted 'severity' instead of 'true_severity' for the bounds
    S_first = float(df['severity'].iloc[0])
    S_last  = float(df['severity'].iloc[-1])

    if decreasing:
        S_max = S_first; S_min = S_last
    else:
        S_min = S_first; S_max = S_last

    S_threshold = S_last
    denom       = S_max - S_min

    if denom <= 0:
        raise ValueError("S_max <= S_min — CSV has no degradation trend.")

    n_days = min(how_many_days, len(df))
    history_df    = df.iloc[:n_days].copy()
    current_day   = int(history_df['current_day'].iloc[-1])
    days_hist     = history_df['current_day'].values.astype(float)
    
    # NEW: Feed the predicted severity history into the curve fitter
    severity_hist = history_df['severity'].values.astype(float)

    smoothed_sev = uniform_filter1d(severity_hist, size=3, mode='nearest')

    if decreasing:
        norm_t_eff = (S_max - smoothed_sev) / denom
    else:
        norm_t_eff = (smoothed_sev - S_min) / denom
    norm_t_eff = np.clip(norm_t_eff, 1e-12, 1.0 - 1e-12)

    def power_law(d, L, alpha):
        return np.clip(d / L, 0.0, 1.0) ** alpha

    p0 = [current_day * 1.5, (ALPHA_LB + ALPHA_UB) / 2.0]
    lb = [current_day + 1,   ALPHA_LB]
    ub = [L_MAX,             ALPHA_UB]

    best_fit, pcov = curve_fit(
        power_law, days_hist, norm_t_eff,
        p0=p0, bounds=(lb, ub), method='trf', maxfev=50_000,
    )
    L_pred     = best_fit[0]
    alpha_pred = best_fit[1]
    perr       = np.sqrt(np.diag(pcov))

    RUL_pred      = L_pred - current_day
    true_last_day = int(df['current_day'].max())
    true_rul      = true_last_day - current_day

    # Reconstruct curves
    true_last_day_plot = int(df['current_day'].max())
    x_max     = int(np.ceil(max(L_pred, true_last_day_plot) + 5))
    plot_days = np.arange(1, x_max + 1, dtype=float)
    norm_fit  = power_law(plot_days, L_pred, alpha_pred)

    if decreasing:
        sev_fit           = S_max - denom * norm_fit
        smoothed_sev_plot = S_max - denom * norm_t_eff
    else:
        sev_fit           = S_min + denom * norm_fit
        smoothed_sev_plot = S_min + denom * norm_t_eff

    return dict(
        fault_class=fault_class,
        fault_name=cfg['name'],
        decreasing=decreasing,
        S_min=S_min, S_max=S_max, S_threshold=S_threshold,
        current_day=current_day,
        days_hist=days_hist,
        severity_hist=severity_hist,
        smoothed_sev_plot=smoothed_sev_plot,
        plot_days=plot_days,
        sev_fit=sev_fit,
        L_pred=L_pred, alpha_pred=alpha_pred,
        L_err=perr[0], alpha_err=perr[1],
        RUL_pred=RUL_pred,
        true_last_day=true_last_day,
        true_rul=true_rul,
        df_full=df,
        n_days=n_days,
    )
    
    
# ─────────────────────────────────────────────────────────────────────────────
# HELPER — load .pkl files (cached)
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    try:
        clf   = joblib.load(PKL_CLASSIFIER)
        reg   = joblib.load(PKL_REGRESSOR)
        scl   = joblib.load(PKL_SCALER)
        return clf, reg, scl, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, str(e)