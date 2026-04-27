"""
app.py  —  Predictive Maintenance Pipeline  |  Streamlit UI
============================================================
Modes:
  • Prediction   — upload CSV → fault class + RUL + deployment plot
  • Validation   — model metrics + classification table + full RUL plot
"""

import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Page background ─────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e8eaf0;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Metric cards ────────────────────────────────────── */
.metric-card {
    background: #1c1f2e;
    border: 1px solid #2e3250;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b92b8;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.82rem;
    color: #6b7299;
    margin-top: 4px;
}

/* ── Fault class badge ───────────────────────────────── */
.fault-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
}
.fault-0 { background:#1e3a2f; color:#4caf50; }
.fault-1 { background:#3a2020; color:#ef5350; }
.fault-2 { background:#3a3020; color:#ffa726; }
.fault-3 { background:#20253a; color:#42a5f5; }

/* ── Section headers ─────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8b92b8;
    border-bottom: 1px solid #2e3250;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

/* ── Tabs ────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-size: 0.95rem;
    font-weight: 600;
    color: #8b92b8;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e8eaf0;
    border-bottom: 2px solid #5c6bc0;
}

/* ── Dataframe / table ───────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Info / error boxes ──────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: #2e3250; }
</style>
""", unsafe_allow_html=True)

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
# HELPER — load .pkl files (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
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
    Replicates rul_prediction.py logic.
    Returns dict with all results, or raises on failure.
    """
    df = rul_df.sort_values('current_day').reset_index(drop=True)

    fault_class = int(df['fault_class'].iloc[0])
    cfg         = FAULT_CONFIG[fault_class]
    decreasing  = cfg['decreasing']

    S_first = float(df['true_severity'].iloc[0])
    S_last  = float(df['true_severity'].iloc[-1])

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
    severity_hist = history_df['true_severity'].values.astype(float)

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
# HELPER — build PREDICTION plot (no ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def build_prediction_plot(r):
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    for spine in ax.spines.values():
        spine.set_color('#2e3250')
    ax.tick_params(colors='#8b92b8')
    ax.xaxis.label.set_color('#8b92b8')
    ax.yaxis.label.set_color('#8b92b8')
    ax.grid(True, linestyle='--', alpha=0.25, color='#2e3250')

    plot_days = r['plot_days']
    sev_fit   = r['sev_fit']
    cur       = r['current_day']

    # Fitted past (solid)
    mask_past = plot_days <= cur
    ax.plot(plot_days[mask_past], sev_fit[mask_past],
            color='#5c6bc0', linewidth=2.4, label='Fitted curve (past)')

    # Extrapolated future (dashed)
    mask_fut = plot_days >= cur
    ax.plot(plot_days[mask_fut], sev_fit[mask_fut],
            color='#5c6bc0', linewidth=2.4, linestyle='--',
            label=f"Extrapolated → RUL = {r['RUL_pred']:.1f} days")

    # Current day marker
    ax.axvline(cur, color='#ce93d8', linestyle=':', linewidth=1.8,
               label=f"Current day ({cur})")

    # Failure threshold
    ax.axhline(r['S_threshold'], color='#ef5350', linestyle='-', linewidth=1.4,
               label=f"Failure threshold ({r['S_threshold']:.3g})")

    # Predicted EOL
    ax.axvline(r['L_pred'], color='#ff8a65', linestyle='--', linewidth=1.4,
               label=f"Predicted EOL (day {r['L_pred']:.1f})")

    ax.set_title(
        f"{r['fault_name']}  —  RUL Prediction\n"
        f"α = {r['alpha_pred']:.3f}  |  Predicted lifecycle = {r['L_pred']:.1f} days",
        color='#e8eaf0', fontsize=12, pad=12
    )
    ax.set_xlabel('Days in Operation', fontsize=11)
    ax.set_ylabel('Fault Severity', fontsize=11)

    if r['decreasing']:
        ax.set_ylim(r['S_min'] * 0.95, r['S_max'] * 1.05)
    else:
        ax.set_ylim(0, r['S_max'] * 1.1)

    ax.legend(loc='best', fontsize=8.5, facecolor='#1c1f2e',
              edgecolor='#2e3250', labelcolor='#c5cae9')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — build VALIDATION plot (full, with true severity)
# ─────────────────────────────────────────────────────────────────────────────
def build_validation_plot(r):
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    for spine in ax.spines.values():
        spine.set_color('#2e3250')
    ax.tick_params(colors='#8b92b8')
    ax.xaxis.label.set_color('#8b92b8')
    ax.yaxis.label.set_color('#8b92b8')
    ax.grid(True, linestyle='--', alpha=0.25, color='#2e3250')

    plot_days = r['plot_days']
    sev_fit   = r['sev_fit']
    cur       = r['current_day']
    df        = r['df_full']

    # Raw scatter
    ax.scatter(r['days_hist'], r['severity_hist'], s=35, color='#ef5350',
               zorder=5, label=f"True severity (days 1–{cur}, used for fit)")

    # Smoothed
    ax.plot(r['days_hist'], r['smoothed_sev_plot'], color='#ffa726',
            linewidth=1.4, linestyle='--', label='Smoothed (3-day avg)')

    # Fitted past
    mask_past = plot_days <= cur
    ax.plot(plot_days[mask_past], sev_fit[mask_past],
            color='#5c6bc0', linewidth=2.2, label='Fitted curve (past)')

    # Extrapolated future
    mask_fut = plot_days >= cur
    ax.plot(plot_days[mask_fut], sev_fit[mask_fut],
            color='#5c6bc0', linewidth=2.2, linestyle='--',
            label=f"Extrapolated → RUL = {r['RUL_pred']:.1f} days")

    # True full lifecycle
    ax.plot(df['current_day'].values, df['true_severity'].values,
            color='#66bb6a', linewidth=1.8, linestyle='-.',
            label='True severity (full lifecycle)')

    # Regressor predicted severity
    ax.plot(df['current_day'].values, df['severity'].values,
            color='#ab47bc', linewidth=1.0, linestyle=':',
            label='Regressor predicted severity (reference)')

    # Current day
    ax.axvline(cur, color='#ce93d8', linestyle=':', linewidth=1.8,
               label=f"Current day ({cur})")

    # Failure threshold
    ax.axhline(r['S_threshold'], color='#ef5350', linestyle='-', linewidth=1.4,
               label=f"Failure threshold ({r['S_threshold']:.3g})")

    # Predicted EOL
    ax.axvline(r['L_pred'], color='#ff8a65', linestyle='--', linewidth=1.4,
               label=f"Predicted EOL (day {r['L_pred']:.1f})")

    # True EOL
    ax.axvline(r['true_last_day'], color='#66bb6a', linestyle=':', linewidth=1.4,
               label=f"True EOL (day {r['true_last_day']})")

    ax.set_title(
        f"{r['fault_name']}  —  Validation RUL Plot\n"
        f"Predicted RUL = {r['RUL_pred']:.1f} days  |  True RUL = {r['true_rul']} days  |  "
        f"Error = {abs(r['RUL_pred'] - r['true_rul']):.1f} days  |  α = {r['alpha_pred']:.3f}",
        color='#e8eaf0', fontsize=11, pad=12
    )
    ax.set_xlabel('Days in Operation', fontsize=11)
    ax.set_ylabel('Fault Severity', fontsize=11)

    if r['decreasing']:
        ax.set_ylim(r['S_min'] * 0.95, r['S_max'] * 1.05)
    else:
        ax.set_ylim(0, r['S_max'] * 1.1)

    ax.legend(loc='best', fontsize=8, facecolor='#1c1f2e',
              edgecolor='#2e3250', labelcolor='#c5cae9')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — fig → streamlit via bytes
# ─────────────────────────────────────────────────────────────────────────────
def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — styled metric card
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ''
    st.markdown(
        f'<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value}</div>'
        f'  {sub_html}'
        f'</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — styled fault badge
# ─────────────────────────────────────────────────────────────────────────────
def fault_badge(fault_class):
    name = FAULT_NAMES.get(fault_class, str(fault_class))
    css  = FAULT_CSS.get(fault_class, "fault-0")
    return f'<span class="fault-badge {css}">{name}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — highlight mismatch rows for fault class table
# ─────────────────────────────────────────────────────────────────────────────
def style_fault_table(df_styled):
    def row_style(row):
        match = row['True FaultClass'] == row['Predicted FaultClass']
        bg    = 'background-color: #1e3a2f' if match else 'background-color: #3a2020'
        return [bg] * len(row)
    return df_styled.apply(row_style, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

# App header
st.markdown("""
<div style="padding: 18px 0 8px 0;">
  <span style="font-size:1.8rem; font-weight:800; letter-spacing:0.03em;">
    ⚙️ Predictive Maintenance Pipeline
  </span>
  <span style="font-size:0.9rem; color:#6b7299; margin-left:16px;">
    Fault Detection  ·  Severity Regression  ·  RUL Estimation
  </span>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load models once
classifier, regressor, scaler, model_err = load_models()

# Tabs
tab_pred, tab_val = st.tabs(["🔮  Prediction", "📊  Model Validation"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred:

    # ── Model load error guard ────────────────────────────────────────────────
    if model_err:
        st.error(f"**Could not load model files.**\n\n{model_err}\n\n"
                 f"Make sure `{PKL_CLASSIFIER}`, `{PKL_REGRESSOR}`, and "
                 f"`{PKL_SCALER}` are in the same directory as `app.py`.")
        st.stop()

    # ── File uploader ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Upload Machine Lifecycle CSV</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload `machine_lifecycle_input.csv`",
        type=["csv"],
        key="pred_upload",
        help="One row per day. Must contain all 14 feature columns, "
             "3 severity columns, FaultClass, and current_day."
    )

    # Day slider — shown only after upload
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.session_state['uploaded_df'] = df_raw

        total_rows = len(df_raw)
        st.markdown('<div class="section-header">Simulation Parameters</div>',
                    unsafe_allow_html=True)

        how_many = st.slider(
            "Current Day (days of data to use for RUL fit)",
            min_value=3,
            max_value=total_rows,
            value=min(20, total_rows),
            step=1,
            help="Simulates 'we are on day N and want to predict remaining life'. "
                 "More days → more accurate fit."
        )

        run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=False)

        if run_btn or ('pred_results' in st.session_state and
                       st.session_state.get('pred_how_many') == how_many and
                       st.session_state.get('pred_file') == uploaded.name):

            if run_btn:
                # Run pipeline
                with st.spinner("Running fault prediction…"):
                    rul_df, final_class, df_pred, fp_err = run_fault_prediction(
                        df_raw.copy(), classifier, regressor, scaler
                    )

                if fp_err:
                    st.error(f"Fault prediction failed: {fp_err}")
                    st.stop()

                if final_class == 0:
                    st.success("✅ Machine is predicted **HEALTHY**. No RUL calculation needed.")
                    st.stop()

                with st.spinner("Fitting degradation curve…"):
                    try:
                        res = run_rul_prediction(rul_df, how_many)
                    except Exception as e:
                        st.error(f"RUL prediction failed: {e}\n\n"
                                 "Try increasing the current day slider or check your CSV.")
                        st.stop()

                st.session_state['pred_results']  = res
                st.session_state['pred_how_many'] = how_many
                st.session_state['pred_file']     = uploaded.name
                st.session_state['pred_df']       = df_pred

            res = st.session_state.get('pred_results')
            if res is None:
                st.info("Press **▶ Run Pipeline** to start.")
            else:
                # ── Result cards ──────────────────────────────────────────────
                st.markdown('<div class="section-header">Prediction Results</div>',
                            unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    badge = fault_badge(res['fault_class'])
                    st.markdown(
                        f'<div class="metric-card">'
                        f'  <div class="metric-label">Detected Fault</div>'
                        f'  <div style="margin-top:10px;">{badge}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c2:
                    metric_card(
                        "Remaining Useful Life",
                        f"{res['RUL_pred']:.1f} days",
                        sub=f"as of day {res['current_day']}"
                    )
                with c3:
                    metric_card(
                        "Predicted Lifecycle",
                        f"{res['L_pred']:.1f} days",
                        sub=f"α = {res['alpha_pred']:.3f}"
                    )
                with c4:
                    metric_card(
                        "Degradation Shape (α)",
                        f"{res['alpha_pred']:.3f}",
                        sub=f"±{res['alpha_err']:.3f}"
                    )

                st.markdown("")

                # ── RUL plot (no ground truth) ────────────────────────────────
                st.markdown('<div class="section-header">RUL Curve</div>',
                            unsafe_allow_html=True)
                fig = build_prediction_plot(res)
                show_fig(fig)

    else:
        st.info("⬆️  Upload a `machine_lifecycle_input.csv` file to begin.")
        if 'pred_results' in st.session_state:
            del st.session_state['pred_results']

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_val:

    # ── Load metrics JSON ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance Metrics</div>',
                unsafe_allow_html=True)

    try:
        with open(METRICS_JSON, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        st.error(f"`{METRICS_JSON}` not found. Run `model_training.py` first to generate it.")
        metrics = None
    except Exception as e:
        st.error(f"Could not load `{METRICS_JSON}`: {e}")
        metrics = None

    if metrics:
        # ── Classifier metrics ────────────────────────────────────────────────
        st.markdown("#### 🌲 Fault Classifier")
        acc = metrics.get('classifier', {}).get('accuracy', None)

        col_acc, col_pad = st.columns([1, 3])
        with col_acc:
            metric_card("Overall Accuracy",
                        f"{acc * 100:.2f}%" if acc is not None else "N/A",
                        sub="RandomForestClassifier")

        st.markdown("")

        # Classification report table
        report = metrics.get('classifier', {}).get('classification_report', {})
        class_labels = ['Healthy', 'LeakFault', 'BlockingFault', 'BearingFault']
        report_rows  = []
        for label in class_labels:
            row_data = report.get(label, {})
            report_rows.append({
                'Class'     : label,
                'Precision' : f"{row_data.get('precision', 0):.3f}",
                'Recall'    : f"{row_data.get('recall', 0):.3f}",
                'F1-Score'  : f"{row_data.get('f1-score', 0):.3f}",
                'Support'   : int(row_data.get('support', 0)),
            })
        df_report = pd.DataFrame(report_rows)
        st.dataframe(df_report, use_container_width=True, hide_index=True)

        st.divider()

        # ── Regressor metrics ─────────────────────────────────────────────────
        st.markdown("#### 📈 Fault Severity Regressor")
        reg_metrics = metrics.get('regressor', {})
        
        # Display Overall Regressor Score (R²)
        overall_r2 = reg_metrics.get('overall_r2', None)
        col_r2, col_pad2 = st.columns([1, 3])
        with col_r2:
            metric_card("Overall R² Score",
                        f"{overall_r2:.4f}" if overall_r2 is not None else "N/A",
                        sub="RandomForestRegressor")
        st.markdown("")

        # Display Per-Column Table
        reg_rows = []
        for col in SEVERITY_COLS:
            m = reg_metrics.get(col, {})
            reg_rows.append({
                'Severity Column' : col,
                'MAE'  : f"{m.get('MAE',  0):.4e}",
                'RMSE' : f"{m.get('RMSE', 0):.4e}",
                'R²'   : f"{m.get('R2',   0):.4f}",
            })
        df_reg = pd.DataFrame(reg_rows)
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

    st.divider()

    # ── Full validation RUL plot ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Validation RUL Plot</div>',
                unsafe_allow_html=True)
    # ... rest of the plotting code remains exactly the same

    st.divider()

    
    # ── Full validation RUL plot ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Validation RUL Plot</div>',
                unsafe_allow_html=True)

    if 'pred_results' in st.session_state:
        res = st.session_state['pred_results']
        st.caption(
            f"Using day **{res['current_day']}** as the cutoff  |  "
            f"True lifecycle: **{res['true_last_day']} days**  |  "
            f"RUL error: **{abs(res['RUL_pred'] - res['true_rul']):.1f} days**"
        )
        fig = build_validation_plot(res)
        show_fig(fig)
    else:
        st.info("Run the pipeline in the **Prediction** tab first to see the validation plot.")
