
"""
app.py — Flask backend for Predictive Maintenance Dashboard
=============================================================
Flow
----
1. POST /upload          — User uploads lifecycle CSV
2. POST /simulate        — MATLAB runs simulation for selected faults
3. POST /predict         — Phase-3 RF models run on uploaded CSV:
                             • classifier → majority-vote fault class (0-7)
                             • regressor  → predicted severity for all 3 columns
                             • power-law curve fit per active fault severity
                             • RUL = min(lifecycle) − current_day across active faults
                             → saves severity-vs-days plot to static/rul_graph.png
4. GET  /validation_graph — Returns rul_graph.png as PNG
5. GET  /static/rul_graph.png — Standard static serving
"""

import traceback
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
MATLAB_DIR = BASE_DIR / "matlab"
PHASE3_DIR = BASE_DIR.parent.parent / "phase_3"

# ── model constants ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    'fPeak':      'Peak Frequency',
    'pLow':       'Low-frequency Power',
    'pMid':       'Mid-frequency Power',
    'pHigh':      'High-frequency Power',
    'pKurtosis':  'Power Kurtosis',
    'qMean':      'Flow Mean',
    'qVar':       'Flow Variance',
    'qSkewness':  'Flow Skewness',
    'qKurtosis':  'Flow Kurtosis',
    'qPeak2Peak': 'Flow Peak-to-Peak',
    'qCrest':     'Flow Crest Factor',
    'qRMS':       'Flow RMS',
    'qMAD':       'Flow Mean Abs. Deviation',
    'qCSRange':   'Cumulative Sum Range',
}

FEATURE_COLS = [
    'fPeak', 'pLow', 'pMid', 'pHigh', 'pKurtosis',
    'qMean', 'qVar', 'qSkewness', 'qKurtosis',
    'qPeak2Peak', 'qCrest', 'qRMS', 'qMAD', 'qCSRange'
]
SEVERITY_COLS = ['LeakFault', 'BlockingFault', 'BearingFault']

FAULT_NAMES = {
    0: 'Healthy',
    1: 'LeakFault',
    2: 'BlockingFault',
    3: 'BearingFault',
    4: 'Leak+Block',
    5: 'Block+Bearing',
    6: 'Bearing+Leak',
    7: 'AllFaults',
}

# Which severity columns are active (and need RUL fitting) per fault class
FAULT_ACTIVE_SEVERITIES = {
    0: [],
    1: ['LeakFault'],
    2: ['BlockingFault'],
    3: ['BearingFault'],
    4: ['LeakFault', 'BlockingFault'],
    5: ['BlockingFault', 'BearingFault'],
    6: ['BearingFault', 'LeakFault'],
    7: ['LeakFault', 'BlockingFault', 'BearingFault'],
}

# Per-severity degradation direction and plot colour
SEVERITY_CONFIG = {
    'LeakFault':     dict(decreasing=False, color='#ef5350'),
    'BlockingFault': dict(decreasing=True,  color='#ffa726'),
    'BearingFault':  dict(decreasing=False, color='#42a5f5'),
}

ALPHA_LB = 1.0
ALPHA_UB = 4.0
L_MAX    = 500

# ── load phase-3 models once at startup ───────────────────────────────────────
try:
    _classifier = joblib.load(PHASE3_DIR / "fault_classifier.pkl")
    _regressor  = joblib.load(PHASE3_DIR / "fault_regressor.pkl")
    _scaler     = joblib.load(PHASE3_DIR / "feature_scaler.pkl")
    _model_err  = None
except Exception as e:
    _classifier = _regressor = _scaler = None
    _model_err  = str(e)

# ── fault → MATLAB script index mapping ───────────────────────────────────────
_FAULT_MAP = {
    frozenset():                                               0,
    frozenset(["LeakFault"]):                                  1,
    frozenset(["BlockingFault"]):                              2,
    frozenset(["BearingFault"]):                               3,
    frozenset(["LeakFault", "BlockingFault"]):                 4,
    frozenset(["BlockingFault", "BearingFault"]):              5,
    frozenset(["BearingFault", "LeakFault"]):                  6,
    frozenset(["LeakFault", "BlockingFault", "BearingFault"]): 7,
}


def faults_to_script_index(faults: list) -> int:
    key = frozenset(faults)
    if key not in _FAULT_MAP:
        raise ValueError(f"Unknown fault combination: {sorted(faults)}")
    return _FAULT_MAP[key]


app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# ── in-memory state ────────────────────────────────────────────────────────────
state = {"lifecycle_df": None}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — run MATLAB simulation
# ═══════════════════════════════════════════════════════════════════════════════
def run_matlab(faults: list) -> None:
    import subprocess
    matlab_path   = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    script_index  = faults_to_script_index(faults)
    matlab_script = MATLAB_DIR / f"run_{script_index}.m"
    if not matlab_script.exists():
        raise FileNotFoundError(f"MATLAB script not found: {matlab_script}")
    script_dir = matlab_script.parent.as_posix()
    command    = f"cd('{script_dir}'); run('{matlab_script.as_posix()}'); exit;"
    subprocess.Popen([matlab_path, "-nosplash", "-r", command]).wait()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — power-law RUL fit for one severity column
# ═══════════════════════════════════════════════════════════════════════════════
def fit_rul_for_severity(days_all: np.ndarray, sev_all: np.ndarray,
                         sev_col: str, up_to_day: float) -> dict:
    cfg        = SEVERITY_CONFIG[sev_col]
    decreasing = cfg['decreasing']

    S_first = float(sev_all[0])
    S_last  = float(sev_all[-1])

    S_max, S_min = (S_first, S_last) if decreasing else (S_last, S_first)
    S_threshold  = S_last
    denom        = S_max - S_min

    if denom <= 0:
        raise ValueError(f"No degradation trend in predicted {sev_col} severity "
                         f"(S_max={S_max:.4e} <= S_min={S_min:.4e}).")

    mask = days_all <= up_to_day
    if mask.sum() < 3:
        raise ValueError(f"Fewer than 3 data points up to day {up_to_day} for {sev_col}.")
    days_hist = days_all[mask].astype(float)
    sev_hist  = sev_all[mask].astype(float)

    smoothed   = uniform_filter1d(sev_hist, size=3, mode='nearest')
    norm_t_eff = ((S_max - smoothed) / denom) if decreasing else ((smoothed - S_min) / denom)
    norm_t_eff = np.clip(norm_t_eff, 1e-12, 1.0 - 1e-12)

    def power_law(d, L, alpha):
        return np.clip(d / L, 0.0, 1.0) ** alpha

    best_fit, _ = curve_fit(
        power_law, days_hist, norm_t_eff,
        p0=[up_to_day * 1.5, (ALPHA_LB + ALPHA_UB) / 2.0],
        bounds=([up_to_day + 1, ALPHA_LB], [L_MAX, ALPHA_UB]),
        method='trf', maxfev=50_000,
    )
    L_pred, alpha_pred = best_fit
    RUL_pred           = L_pred - up_to_day

    x_max      = int(np.ceil(max(L_pred, days_all[-1]) + 5))
    plot_days  = np.arange(1, x_max + 1, dtype=float)
    norm_fit   = power_law(plot_days, L_pred, alpha_pred)
    sev_fit    = (S_max - denom * norm_fit) if decreasing else (S_min + denom * norm_fit)

    return {
        'L_pred':      L_pred,
        'alpha_pred':  alpha_pred,
        'RUL_pred':    RUL_pred,
        'current_day': up_to_day,
        'S_threshold': S_threshold,
        'S_min':       S_min,
        'S_max':       S_max,
        'decreasing':  decreasing,
        'days_hist':   days_hist,
        'sev_hist':    sev_hist,
        'plot_days':   plot_days,
        'sev_fit':     sev_fit,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — severity vs days plot for all active faults
# ═══════════════════════════════════════════════════════════════════════════════
def generate_rul_plot(rul_results: dict, fault_name: str) -> None:
    n = len(rul_results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.5 * n),
                             facecolor='#0f1117', squeeze=False)

    for ax, (sev_col, r) in zip(axes[:, 0], rul_results.items()):
        color = SEVERITY_CONFIG[sev_col]['color']
        ax.set_facecolor('#0f1117')
        for spine in ax.spines.values():
            spine.set_color('#2e3250')
        ax.tick_params(colors='#8b92b8')
        ax.xaxis.label.set_color('#8b92b8')
        ax.yaxis.label.set_color('#8b92b8')
        ax.grid(True, linestyle='--', alpha=0.25, color='#2e3250')

        ax.scatter(r['days_hist'], r['sev_hist'], s=30, color=color,
                   zorder=5, alpha=0.7, label='Predicted severity (history)')

        mask_past = r['plot_days'] <= r['current_day']
        ax.plot(r['plot_days'][mask_past], r['sev_fit'][mask_past],
                color=color, linewidth=2.2, label='Fitted curve')

        mask_fut = r['plot_days'] >= r['current_day']
        ax.plot(r['plot_days'][mask_fut], r['sev_fit'][mask_fut],
                color=color, linewidth=2.2, linestyle='--',
                label=f"Extrapolated → RUL = {r['RUL_pred']:.1f} days")

        ax.axvline(r['current_day'], color='#ce93d8', linestyle=':', linewidth=1.8,
                   label=f"Current day ({int(r['current_day'])})")
        ax.axhline(r['S_threshold'], color='#ef5350', linestyle='-', linewidth=1.4,
                   label=f"Failure threshold ({r['S_threshold']:.3g})")
        ax.axvline(r['L_pred'], color='#ff8a65', linestyle='--', linewidth=1.4,
                   label=f"Predicted EOL (day {r['L_pred']:.1f})")

        ax.set_title(
            f"{sev_col}  |  Lifecycle = {r['L_pred']:.1f} days  |  α = {r['alpha_pred']:.3f}",
            color='#e8eaf0', fontsize=11, pad=8,
        )
        ax.set_xlabel('Days in Operation', fontsize=10)
        ax.set_ylabel('Fault Severity', fontsize=10)
        ylim_lo = r['S_min'] * 0.95 if r['decreasing'] else 0
        ylim_hi = r['S_max'] * 1.05 if r['decreasing'] else r['S_max'] * 1.1
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.legend(loc='best', fontsize=8, facecolor='#1c1f2e',
                  edgecolor='#2e3250', labelcolor='#c5cae9')

    fig.suptitle(f'RUL Prediction — {fault_name}', color='#e8eaf0', fontsize=13)
    plt.tight_layout()
    fig.savefig(str(STATIC_DIR / "rul_graph.png"), dpi=120,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — validation plot: predicted + real severity overlaid, RUL comparison
# ═══════════════════════════════════════════════════════════════════════════════
def generate_validation_plot(rul_results: dict, fault_name: str,
                              df: pd.DataFrame, days_all: np.ndarray,
                              up_to_day: float, final_rul: float,
                              dominant_col: str = None):
    """
    Like generate_rul_plot but adds real severity values from the CSV and marks
    the real end-of-life from total_lifecycle_days.
    Returns (real_lifecycle, real_rul, rul_error) or (None, None, None).
    """
    real_lifecycle = None
    if 'total_lifecycle_days' in df.columns:
        real_lifecycle = float(df['total_lifecycle_days'].iloc[0])
    real_rul   = (real_lifecycle - up_to_day) if real_lifecycle is not None else None
    rul_error  = (final_rul - real_rul)       if real_rul   is not None else None

    # Show only the dominant fault subplot in the validation view
    if dominant_col and dominant_col in rul_results:
        rul_results = {dominant_col: rul_results[dominant_col]}

    n = len(rul_results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.5 * n),
                             facecolor='#0f1117', squeeze=False)

    for ax, (sev_col, r) in zip(axes[:, 0], rul_results.items()):
        color = SEVERITY_CONFIG[sev_col]['color']
        ax.set_facecolor('#0f1117')
        for spine in ax.spines.values():
            spine.set_color('#2e3250')
        ax.tick_params(colors='#8b92b8')
        ax.xaxis.label.set_color('#8b92b8')
        ax.yaxis.label.set_color('#8b92b8')
        ax.grid(True, linestyle='--', alpha=0.25, color='#2e3250')

        # ── Real severity (all CSV rows, crosses) ──────────────────────────
        if sev_col in df.columns:
            ax.scatter(days_all, df[sev_col].values, s=35, color='#66bb6a',
                       marker='x', linewidths=1.5, zorder=7, alpha=0.85,
                       label='Real severity (ground truth)')

        # ── Predicted severity scatter (history up to current_day) ─────────
        ax.scatter(r['days_hist'], r['sev_hist'], s=30, color=color,
                   zorder=5, alpha=0.7, label='Predicted severity')

        # ── Power-law fit: past ────────────────────────────────────────────
        mask_past = r['plot_days'] <= r['current_day']
        ax.plot(r['plot_days'][mask_past], r['sev_fit'][mask_past],
                color=color, linewidth=2.2, label='Fitted curve (predicted)')

        # ── Power-law fit: extrapolated future ────────────────────────────
        mask_fut = r['plot_days'] >= r['current_day']
        ax.plot(r['plot_days'][mask_fut], r['sev_fit'][mask_fut],
                color=color, linewidth=2.2, linestyle='--',
                label=f"Extrapolated → Pred. RUL = {r['RUL_pred']:.1f} d")

        # ── Current day ───────────────────────────────────────────────────
        ax.axvline(r['current_day'], color='#ce93d8', linestyle=':', linewidth=1.8,
                   label=f"Current day ({int(r['current_day'])})")

        # ── Failure threshold ─────────────────────────────────────────────
        ax.axhline(r['S_threshold'], color='#ef5350', linestyle='-', linewidth=1.4,
                   label=f"Failure threshold ({r['S_threshold']:.3g})")

        # ── Predicted EOL ─────────────────────────────────────────────────
        ax.axvline(r['L_pred'], color='#ff8a65', linestyle='--', linewidth=1.4,
                   label=f"Predicted EOL (day {r['L_pred']:.1f})")

        # ── Real EOL ──────────────────────────────────────────────────────
        if real_lifecycle is not None:
            ax.axvline(real_lifecycle, color='#66bb6a', linestyle='--', linewidth=1.8,
                       label=f"Real EOL (day {int(real_lifecycle)})")

        ax.set_title(
            f"{sev_col}  |  Predicted EOL = {r['L_pred']:.1f} d  |  α = {r['alpha_pred']:.3f}",
            color='#e8eaf0', fontsize=11, pad=8,
        )
        ax.set_xlabel('Days in Operation', fontsize=10)
        ax.set_ylabel('Fault Severity', fontsize=10)
        ylim_lo = r['S_min'] * 0.95 if r['decreasing'] else 0
        ylim_hi = r['S_max'] * 1.05 if r['decreasing'] else r['S_max'] * 1.1
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.legend(loc='best', fontsize=8, facecolor='#1c1f2e',
                  edgecolor='#2e3250', labelcolor='#c5cae9')

    if real_lifecycle is not None:
        title = (
            f'Validation — {fault_name}\n'
            f'Predicted RUL: {final_rul:.1f} d  ·  '
            f'Real RUL: {real_rul:.1f} d  ·  '
            f'Error: {rul_error:+.1f} d'
        )
    else:
        title = f'Validation — {fault_name}  |  Predicted RUL: {final_rul:.1f} d'

    fig.suptitle(title, color='#e8eaf0', fontsize=12)
    plt.tight_layout()
    fig.savefig(str(STATIC_DIR / "validation_graph.png"), dpi=120,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close(fig)

    return real_lifecycle, real_rul, rul_error


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE — POST /upload
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    state["lifecycle_df"] = df

    preview = df.head(100)
    records = [
        {col: (round(float(v), 4) if hasattr(v, "__float__") and str(v) not in ("nan", "inf")
               else str(v))
         for col, v in row.items()}
        for _, row in preview.iterrows()
    ]

    return jsonify({
        "message":    f"Uploaded {len(df)} rows",
        "total_rows": len(df),
        "dataset":    records,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE — POST /simulate
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/simulate", methods=["POST"])
def simulate():
    body   = request.get_json(force=True)
    faults = body.get("faults", [])
    try:
        run_matlab(faults)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"MATLAB failed: {str(e)}"}), 500
    return jsonify({"message": "Simulation complete"})


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE — POST /predict
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    if state["lifecycle_df"] is None:
        return jsonify({"error": "No dataset. Upload a lifecycle CSV first."}), 400
    if _model_err:
        return jsonify({"error": f"Models failed to load: {_model_err}"}), 500

    body       = request.get_json(force=True)
    up_to_day  = float(body.get("currentDay", 0))

    df = state["lifecycle_df"].copy()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"CSV missing required feature columns: {missing}"}), 400

    if 'current_day' in df.columns:
        df = df.sort_values('current_day').reset_index(drop=True)
        days_all = df['current_day'].values.astype(float)
    else:
        days_all = np.arange(1, len(df) + 1, dtype=float)

    if up_to_day <= 0:
        up_to_day = float(days_all[-1])

    X_scaled    = _scaler.transform(df[FEATURE_COLS].values)
    pred_classes = _classifier.predict(X_scaled)
    final_class  = int(Counter(pred_classes).most_common(1)[0][0])
    fault_name   = FAULT_NAMES.get(final_class, str(final_class))
    active_cols  = FAULT_ACTIVE_SEVERITIES.get(final_class, [])

    if not active_cols:
        return jsonify({
            "fault":       fault_name,
            "rul":         None,
            "current_day": int(up_to_day),
        })

    pred_sev = _regressor.predict(X_scaled)   # shape (n_rows, 3)
    for i, col in enumerate(SEVERITY_COLS):
        df[f'pred_{col}'] = pred_sev[:, i]

    rul_results = {}
    for sev_col in active_cols:
        sev_all = df[f'pred_{sev_col}'].values
        try:
            rul_results[sev_col] = fit_rul_for_severity(days_all, sev_all, sev_col, up_to_day)
        except Exception as e:
            return jsonify({"error": f"RUL fit failed for {sev_col}: {e}"}), 500

    # Worst-case fault drives the RUL
    worst_col   = min(rul_results, key=lambda c: rul_results[c]['L_pred'])
    final_rul   = rul_results[worst_col]['RUL_pred']
    current_day = int(up_to_day)

    try:
        generate_rul_plot(rul_results, fault_name)
    except Exception:
        traceback.print_exc()

    real_lifecycle = real_rul = rul_error = None
    try:
        real_lifecycle, real_rul, rul_error = generate_validation_plot(
            rul_results, fault_name, df, days_all, up_to_day, final_rul,
            dominant_col=worst_col
        )
    except Exception:
        traceback.print_exc()

    response = {
        "fault":       fault_name,
        "rul":         round(final_rul, 2),
        "current_day": current_day,
    }
    if real_rul is not None:
        response["real_rul"]         = round(real_rul, 2)
        response["rul_error"]        = round(rul_error, 2)
        response["real_lifecycle"]   = int(real_lifecycle)
    return jsonify(response)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE — GET /validation_graph
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/validation_graph", methods=["GET"])
def validation_graph():
    graph_path = STATIC_DIR / "validation_graph.png"
    if not graph_path.exists():
        return jsonify({"error": "No validation graph yet. Run prediction first."}), 404
    return send_file(str(graph_path), mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE — POST /explain
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/explain", methods=["POST"])
def explain():
    if state["lifecycle_df"] is None:
        return jsonify({"error": "No dataset. Upload a lifecycle CSV first."}), 400
    if _model_err:
        return jsonify({"error": f"Models failed to load: {_model_err}"}), 500

    try:
        import shap
    except ImportError:
        return jsonify({"error": "SHAP not installed. Run: pip install shap"}), 500

    df = state["lifecycle_df"].copy()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"CSV missing columns: {missing}"}), 400

    if 'current_day' in df.columns:
        df = df.sort_values('current_day').reset_index(drop=True)

    X_scaled = _scaler.transform(df[FEATURE_COLS].values)
    X_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

    # Determine fault class from classifier
    pred_classes = _classifier.predict(X_scaled)
    final_class  = int(Counter(pred_classes).most_common(1)[0][0])
    fault_name   = FAULT_NAMES.get(final_class, str(final_class))
    active_cols  = FAULT_ACTIVE_SEVERITIES.get(final_class, [])

    try:
        explainer = shap.TreeExplainer(_regressor)
        shap_vals = explainer.shap_values(X_df)
        # SHAP returns (n_samples, n_features, n_outputs) for multi-output RF
        if isinstance(shap_vals, list):
            shap_3d = np.stack(shap_vals, axis=-1)  # (n_samples, n_features, n_outputs)
        else:
            shap_3d = np.array(shap_vals)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"SHAP computation failed: {str(e)}"}), 500

    candidates = active_cols if active_cols else SEVERITY_COLS
    top_k      = 6

    explanations = []
    for col in candidates:
        idx        = SEVERITY_COLS.index(col)
        sv         = shap_3d[:, :, idx]          # (n_samples, n_features)
        mean_abs   = np.mean(np.abs(sv), axis=0) # (n_features,)
        mean_signed= np.mean(sv, axis=0)         # (n_features,)

        # BlockingFault severity DECREASES as fault worsens (0.8 → 0.53), so
        # features that push it DOWN are the degradation drivers.
        # LeakFault and BearingFault severity INCREASE, so features pushing UP matter.
        decreasing = SEVERITY_CONFIG[col]['decreasing']
        if decreasing:
            degrad_mask = mean_signed < 0   # BlockingFault: show features pushing DOWN
        else:
            degrad_mask = mean_signed > 0   # LeakFault, BearingFault: show features pushing UP

        # Rank only degradation-direction features; fall back to all if none qualify
        filtered_abs = np.where(degrad_mask, mean_abs, 0.0)
        if filtered_abs.sum() == 0:
            filtered_abs = mean_abs          # fallback: nothing filtered, keep all
        total_abs = float(filtered_abs.sum()) or 1.0
        order = [i for i in np.argsort(filtered_abs)[::-1] if filtered_abs[i] > 0][:top_k]

        features = []
        for i in order:
            feat  = FEATURE_COLS[i]
            signed= float(mean_signed[i])
            abs_v = float(filtered_abs[i])
            pct   = round(abs_v / total_abs * 100, 1)
            label = FEATURE_LABELS.get(feat, feat)
            if decreasing:
                direction = "down"
                text = f"{label} decreased in this data, driving {col} severity lower — indicating worsening blockage."
            else:
                direction = "up"
                text = f"{label} increased in this data, pushing {col} severity higher — indicating fault progression."
            features.append({
                "feature":          feat,
                "label":            label,
                "contribution":     round(signed, 4),
                "abs_contribution": round(abs_v, 4),
                "pct":              pct,
                "direction":        direction,
                "text":             text,
            })

        top_label = features[0]["label"] if features else "unknown"
        if decreasing:
            sev_trend = "lower (fault worsening)"
        else:
            sev_trend = "higher (fault progression)"
        explanations.append({
            "severity": col,
            "features": features,
            "summary":  (
                f"{top_label} had the strongest influence on {col} severity, "
                f"driving it {sev_trend} across the lifecycle."
            ),
        })

    overall_summary = (
        f"The model identified {fault_name}. "
        f"SHAP explanations are shown below for "
        f"{'each active fault severity' if len(candidates) > 1 else 'the active fault severity'} "
        f"({', '.join(candidates)})."
    )

    return jsonify({
        "fault":        fault_name,
        "explanations": explanations,
        "summary":      overall_summary,
    })


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=5000)
