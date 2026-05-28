"""
classify_single.py
-------------------
Upload a single-row CSV → runs fault classifier + regressor → displays results.

CSV format expected:
    14 feature cols + LeakFault + BlockingFault + BearingFault + FaultClass
    (severity and FaultClass columns are ignored by the models — reference only)

Run with:
    streamlit run classify_single.py
"""

import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fault Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; color: #e8eaf0; }
[data-testid="stHeader"]           { background: transparent; }

.metric-card {
    background: #1c1f2e;
    border: 1px solid #2e3250;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b92b8;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1.15;
}
.metric-sub {
    font-size: 0.8rem;
    color: #6b7299;
    margin-top: 4px;
}
.fault-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
}
.fault-0 { background:#1e3a2f; color:#4caf50; }
.fault-1 { background:#3a2020; color:#ef5350; }
.fault-2 { background:#3a3020; color:#ffa726; }
.fault-3 { background:#20253a; color:#42a5f5; }

.section-header {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #8b92b8;
    border-bottom: 1px solid #2e3250;
    padding-bottom: 6px;
    margin: 24px 0 14px 0;
}
hr { border-color: #2e3250; }
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'fPeak', 'pLow', 'pMid', 'pHigh', 'pKurtosis',
    'qMean', 'qVar', 'qSkewness', 'qKurtosis',
    'qPeak2Peak', 'qCrest', 'qRMS', 'qMAD', 'qCSRange'
]
SEVERITY_COLS   = ['LeakFault', 'BlockingFault', 'BearingFault']
FAULT_NAMES     = {0: 'Healthy', 1: 'LeakFault', 2: 'BlockingFault', 3: 'BearingFault'}
FAULT_CSS       = {0: 'fault-0', 1: 'fault-1', 2: 'fault-2', 3: 'fault-3'}

PKL_CLASSIFIER  = "fault_classifier.pkl"
PKL_REGRESSOR   = "fault_regressor.pkl"
PKL_SCALER      = "feature_scaler.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        clf = joblib.load(PKL_CLASSIFIER)
        reg = joblib.load(PKL_REGRESSOR)
        scl = joblib.load(PKL_SCALER)
        return clf, reg, scl, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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

def fault_badge(fault_class):
    name = FAULT_NAMES.get(fault_class, str(fault_class))
    css  = FAULT_CSS.get(fault_class, 'fault-0')
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">Predicted Fault Class</div>'
        f'  <div style="margin-top:12px;">'
        f'    <span class="fault-badge {css}">{name}</span>'
        f'  </div>'
        f'</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 16px 0 6px 0;">
  <span style="font-size:1.6rem; font-weight:800;">🔬 Fault Classifier</span>
  <span style="font-size:0.88rem; color:#6b7299; margin-left:14px;">
    Single-row inference  ·  Classifier + Regressor
  </span>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Model load check ──────────────────────────────────────────────────────────
classifier, regressor, scaler, model_err = load_models()
if model_err:
    st.error(
        f"**Could not load model files.**\n\n{model_err}\n\n"
        f"Make sure `{PKL_CLASSIFIER}`, `{PKL_REGRESSOR}`, and `{PKL_SCALER}` "
        f"are in the same directory as this script."
    )
    st.stop()

# ── File upload ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Upload CSV</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a single-row feature CSV",
    type=["csv"],
    help="Must contain the 14 feature columns. "
         "LeakFault, BlockingFault, BearingFault, and FaultClass are optional — "
         "they are ignored by the models and shown only as reference."
)

if uploaded is None:
    st.info("⬆️  Upload a CSV file to run classification.")
    st.stop()

# ── Parse CSV ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if len(df) == 0:
    st.error("The uploaded CSV is empty.")
    st.stop()

if len(df) > 1:
    st.warning(f"CSV has {len(df)} rows — only the first row will be used.")
    df = df.iloc[[0]]

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error(f"CSV is missing required feature columns: `{missing}`")
    st.stop()

# ── Run models ────────────────────────────────────────────────────────────────
X        = df[FEATURE_COLS]
X_scaled = scaler.transform(X)

pred_class    = int(classifier.predict(X_scaled)[0])
pred_severity = regressor.predict(X_scaled)[0]   # shape (3,)

pred_leak     = pred_severity[0]
pred_block    = pred_severity[1]
pred_bearing  = pred_severity[2]

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Classification Result</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(fault_badge(pred_class), unsafe_allow_html=True)
with c2:
    metric_card("Predicted LeakFault Severity",    f"{pred_leak:.4e}")
with c3:
    metric_card("Predicted BlockingFault Severity", f"{pred_block:.4f}")
with c4:
    metric_card("Predicted BearingFault Severity",  f"{pred_bearing:.4e}")

# ── Reference values (if present in CSV) ─────────────────────────────────────
has_ref = all(c in df.columns for c in SEVERITY_COLS + ['FaultClass'])
if has_ref:
    st.markdown('<div class="section-header">Reference Values (from CSV)</div>',
                unsafe_allow_html=True)

    true_class = int(df['FaultClass'].iloc[0])
    match      = pred_class == true_class
    match_html = (
        '<span style="color:#4caf50; font-weight:700;">✓ Correct</span>'
        if match else
        '<span style="color:#ef5350; font-weight:700;">✗ Incorrect</span>'
    )

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    with rc1:
        st.markdown(
            f'<div class="metric-card">'
            f'  <div class="metric-label">True Fault Class</div>'
            f'  <div style="margin-top:10px;">'
            f'    <span class="fault-badge {FAULT_CSS.get(true_class, "fault-0")}">'
            f'      {FAULT_NAMES.get(true_class, str(true_class))}'
            f'    </span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with rc2:
        st.markdown(
            f'<div class="metric-card">'
            f'  <div class="metric-label">Classification</div>'
            f'  <div class="metric-value" style="font-size:1.4rem; margin-top:8px;">'
            f'    {match_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with rc3:
        metric_card("True LeakFault Severity",
                    f"{df['LeakFault'].iloc[0]:.4e}")
    with rc4:
        metric_card("True BlockingFault Severity",
                    f"{df['BlockingFault'].iloc[0]:.4f}")
    with rc5:
        metric_card("True BearingFault Severity",
                    f"{df['BearingFault'].iloc[0]:.4e}")

# ── Raw feature values ────────────────────────────────────────────────────────
with st.expander("Show input feature values"):
    st.dataframe(df[FEATURE_COLS].T.rename(columns={df.index[0]: 'Value'}),
                 use_container_width=True)
