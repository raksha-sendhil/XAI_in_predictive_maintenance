"""
explanations.py
---------------
Streamlit app for:
1) Uploading lifecycle test CSV
2) Running classifier + regressor predictions (same core logic as app.py)
3) Showing predicted vs real table + majority-vote final fault class
4) Explaining one selected day using SHAP for classifier and regressor

Run:
    streamlit run explanations.py
"""

import warnings
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import shap
except Exception:
    shap = None


FEATURE_COLS = [
    "fPeak", "pLow", "pMid", "pHigh", "pKurtosis",
    "qMean", "qVar", "qSkewness", "qKurtosis",
    "qPeak2Peak", "qCrest", "qRMS", "qMAD", "qCSRange",
]
SEVERITY_COLS = ["LeakFault", "BlockingFault", "BearingFault"]
FAULT_CLASS_MAP = {0: None, 1: "LeakFault", 2: "BlockingFault", 3: "BearingFault"}
FAULT_NAMES = {0: "Healthy", 1: "LeakFault", 2: "BlockingFault", 3: "BearingFault"}

PKL_CLASSIFIER = "fault_classifier.pkl"
PKL_REGRESSOR = "fault_regressor.pkl"
PKL_SCALER = "feature_scaler.pkl"


@st.cache_resource(show_spinner=False)
def load_models():
    clf = joblib.load(PKL_CLASSIFIER)
    reg = joblib.load(PKL_REGRESSOR)
    scl = joblib.load(PKL_SCALER)
    return clf, reg, scl


def run_predictions(df_input, classifier, regressor, scaler):
    required = FEATURE_COLS + SEVERITY_COLS + ["FaultClass", "current_day"]
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df_input.copy()
    x = df[FEATURE_COLS]
    x_scaled = scaler.transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=FEATURE_COLS, index=df.index)

    pred_class = classifier.predict(x_scaled)
    pred_sev_all = regressor.predict(x_scaled)

    df["pred_FaultClass"] = pred_class
    for idx, col in enumerate(SEVERITY_COLS):
        df[f"pred_{col}"] = pred_sev_all[:, idx]

    counts = Counter(pred_class)
    final_class = counts.most_common(1)[0][0]
    final_sev_col = FAULT_CLASS_MAP.get(final_class)

    if final_sev_col is None:
        df["pred_final_severity"] = np.nan
        df["true_final_severity"] = np.nan
    else:
        df["pred_final_severity"] = df[f"pred_{final_sev_col}"]
        df["true_final_severity"] = df[final_sev_col]

    view_df = df[[
        "current_day",
        "pred_final_severity",
        "true_final_severity",
        "pred_FaultClass",
        "FaultClass",
    ]].rename(columns={
        "pred_final_severity": "Predicted Severity",
        "true_final_severity": "Real Severity",
        "pred_FaultClass": "Predicted Fault Class",
        "FaultClass": "Real Fault Class",
    })

    return df, x_scaled_df, final_class, final_sev_col, view_df


def top_feature_text(shap_vals, row_scaled, feature_names, top_k=6):
    order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    lines = []
    for i in order:
        feat = feature_names[i]
        impact = shap_vals[i]
        direction = "increases" if impact > 0 else "decreases"
        strength = "strongly" if abs(impact) > np.percentile(np.abs(shap_vals), 70) else "slightly"
        lines.append(
            f"- `{feat}` ({row_scaled[i]:.3f}) {strength} {direction} this model output."
        )
    return lines


def shap_summary_bar(shap_values_row, row_df, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(
        shap.Explanation(
            values=shap_values_row.values,
            base_values=shap_values_row.base_values,
            data=row_df.values[0],
            feature_names=list(row_df.columns),
        ),
        max_display=10,
        show=False,
    )
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


st.set_page_config(page_title="SHAP Explanations", layout="wide")
st.title("Model Explanations (Classifier + Regressor)")
st.caption("Upload lifecycle CSV, run predictions, then inspect SHAP explanation for any day.")

try:
    classifier, regressor, scaler = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload test CSV (e.g., machine_lifecycle_input.csv / input2 / input3)",
    type=["csv"],
)

if uploaded is None:
    st.info("Upload a CSV to continue.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if len(raw_df) == 0:
    st.error("Uploaded CSV is empty.")
    st.stop()

try:
    pred_df, x_scaled_df, final_class, final_sev_col, result_table = run_predictions(
        raw_df, classifier, regressor, scaler
    )
except Exception as e:
    st.error(str(e))
    st.stop()

left, right = st.columns(2)
with left:
    st.metric("Final Fault Class (Majority Vote)", FAULT_NAMES.get(final_class, str(final_class)))
with right:
    st.metric("Rows in Uploaded CSV", len(pred_df))

st.subheader("Predicted vs Real Table")
st.dataframe(result_table.sort_values("current_day"), use_container_width=True, hide_index=True)

if shap is None:
    st.warning("SHAP is not installed in this environment. Install with: pip install shap")
    st.stop()

day_vals = sorted(pred_df["current_day"].astype(int).unique().tolist())
selected_day = st.slider(
    "Pick a day number for explanation",
    min_value=min(day_vals),
    max_value=max(day_vals),
    value=day_vals[min(2, len(day_vals) - 1)],
)

day_mask = pred_df["current_day"].astype(int) == int(selected_day)
if not day_mask.any():
    st.error("No row found for selected day.")
    st.stop()

row_idx = pred_df[day_mask].index[0]
row_scaled_df = x_scaled_df.loc[[row_idx]]
row_scaled = row_scaled_df.values[0]

st.subheader(f"Explanation for Day {selected_day}")
st.write(
    "Below is what pushed the model outputs up or down for this day, in plain language."
)

# Classifier SHAP
clf_explainer = shap.TreeExplainer(classifier)
clf_sv = clf_explainer(row_scaled_df)
pred_class_this_row = int(pred_df.loc[row_idx, "pred_FaultClass"])

if len(clf_sv.values.shape) == 3:
    class_index = pred_class_this_row
    clf_row_exp = shap.Explanation(
        values=clf_sv.values[0, :, class_index],
        base_values=clf_sv.base_values[0, class_index],
        data=row_scaled_df.values[0],
        feature_names=FEATURE_COLS,
    )
else:
    class_index = pred_class_this_row
    clf_row_exp = shap.Explanation(
        values=clf_sv[class_index].values[0],
        base_values=clf_sv[class_index].base_values[0],
        data=row_scaled_df.values[0],
        feature_names=FEATURE_COLS,
    )

st.markdown(f"**Classifier output explained:** `{FAULT_NAMES.get(pred_class_this_row, pred_class_this_row)}`")
shap_summary_bar(clf_row_exp, row_scaled_df, "Top classifier feature impacts")
for line in top_feature_text(clf_row_exp.values, row_scaled, FEATURE_COLS):
    st.write(line)

# Regressor SHAP
reg_explainer = shap.TreeExplainer(regressor)
reg_sv = reg_explainer(row_scaled_df)

if final_class == 0:
    st.markdown("**Regressor output explained:** machine predicted Healthy, so showing `LeakFault` severity output.")
    sev_target_idx = 0
    sev_label = "LeakFault"
else:
    sev_target_idx = SEVERITY_COLS.index(final_sev_col)
    sev_label = final_sev_col

if len(reg_sv.values.shape) == 3:
    reg_row_exp = shap.Explanation(
        values=reg_sv.values[0, :, sev_target_idx],
        base_values=reg_sv.base_values[0, sev_target_idx],
        data=row_scaled_df.values[0],
        feature_names=FEATURE_COLS,
    )
else:
    reg_row_exp = shap.Explanation(
        values=reg_sv[sev_target_idx].values[0],
        base_values=reg_sv[sev_target_idx].base_values[0],
        data=row_scaled_df.values[0],
        feature_names=FEATURE_COLS,
    )

st.markdown(f"**Regressor output explained:** `{sev_label}` severity prediction")
shap_summary_bar(reg_row_exp, row_scaled_df, f"Top regressor impacts for {sev_label}")
for line in top_feature_text(reg_row_exp.values, row_scaled, FEATURE_COLS):
    st.write(line)

st.info(
    "Reading tip: 'increases' means it pushed the model output higher for this specific day; "
    "'decreases' means it pushed it lower."
)

