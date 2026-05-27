
"""
app.py — Flask backend for Predictive Maintenance Dashboard
=============================================================
Flow
----
1. POST /upload          — User uploads lifecycle CSV (e.g. days 1-39)
2. POST /simulate        — MATLAB runs ONE day simulation for the selected faults
                           → MATLAB output is appended as the LAST row to the
                             uploaded lifecycle CSV → merged CSV stored in memory
3. POST /predict         — ML models predict RUL on the FULL merged lifecycle
                             (days 1-39 from CSV  +  day 40 from MATLAB)
4. GET  /validation_graph — Returns actual-vs-predicted RUL plot as PNG
5. GET  /static/rul_graph.png — Standard RUL graph (served by Flask static)
"""
 
import io
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
 
MATLAB_SCRIPT = BASE_DIR / "run_simulation.m"   # your existing MATLAB script
MATLAB_OUT    = BASE_DIR / "matlab_output.csv"  # MATLAB writes its row(s) here
 
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)
 
# ── in-memory state ────────────────────────────────────────────────────────
state = {
    "lifecycle_df":  None,   # uploaded CSV (days 1..N-1)
    "matlab_df":     None,   # MATLAB-produced last day
    "merged_df":     None,   # lifecycle_df + matlab_df  (full lifecycle)
    "model_clf":     None,   # fault classifier
    "model_reg":     None,   # RUL regressor
    "label_enc":     None,   # LabelEncoder for fault column
    "feature_cols":  None,   # column names used for training
}
 
 
# ═══════════════════════════════════════════════════════════════════════════
# HELPER — run MATLAB
# ═══════════════════════════════════════════════════════════════════════════
def run_matlab(faults: list[str]) -> pd.DataFrame:

    matlab_path = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"

    faults_cell = "{'" + "','".join(faults) + "'}"

    command = (
        f"faults={faults_cell}; "
        f"run('{MATLAB_SCRIPT.as_posix()}'); "
        f"exit;"
    )

    process = subprocess.Popen([
        matlab_path,
        "-nosplash",
        "-r",
        command
    ])

    # wait until matlab closes
    process.wait()

    if not MATLAB_OUT.exists():
        raise FileNotFoundError(
            f"MATLAB output file not found: {MATLAB_OUT}"
        )

    df = pd.read_csv(MATLAB_OUT)

    return df
 
 
# ═══════════════════════════════════════════════════════════════════════════
# HELPER — train / retrain models on merged data
# ═══════════════════════════════════════════════════════════════════════════
def train_models(df: pd.DataFrame):
    """
    Trains RandomForest classifier (fault type) and regressor (RUL).
    Expects columns:  <sensor cols...>,  fault,  rul
    Adjust feature_cols to match YOUR CSV column names.
    """
    df = df.copy().dropna()
 
    # Detect feature columns = everything except 'fault' and 'rul'
    exclude = {"fault", "rul", "day", "time", "timestamp"}
    feature_cols = [c for c in df.columns if c.lower() not in exclude]
 
    le = LabelEncoder()
    df["fault_enc"] = le.fit_transform(df["fault"].astype(str))
 
    X = df[feature_cols].values
    y_cls = df["fault_enc"].values
    y_reg = df["rul"].values
 
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )
 
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, yc_tr)
 
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_tr, yr_tr)
 
    state["model_clf"]    = clf
    state["model_reg"]    = reg
    state["label_enc"]    = le
    state["feature_cols"] = feature_cols
 
    return clf, reg, le, feature_cols
 
 
# ═══════════════════════════════════════════════════════════════════════════
# HELPER — generate RUL graph and save to static/
# ═══════════════════════════════════════════════════════════════════════════
def generate_rul_graph(df: pd.DataFrame, predicted_rul_series: np.ndarray):
    days = np.arange(1, len(df) + 1)
    actual_rul = df["rul"].values if "rul" in df.columns else np.zeros(len(df))
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, actual_rul,          color="#22d3ee", linewidth=2,   label="Actual RUL")
    ax.plot(days, predicted_rul_series, color="#f59e0b", linewidth=2,
            linestyle="--", label="Predicted RUL")
 
    # Highlight the last (MATLAB-added) day
    ax.axvline(x=days[-1], color="#f43f5e", linestyle=":", linewidth=1.5,
               label=f"MATLAB Day ({days[-1]})")
 
    ax.set_xlabel("Operating Day")
    ax.set_ylabel("RUL (days)")
    ax.set_title("Remaining Useful Life — Full Lifecycle")
    ax.legend()
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
 
    fig.tight_layout()
    fig.savefig(str(STATIC_DIR / "rul_graph.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ROUTE — POST /upload
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts the lifecycle CSV (days 1..N-1).
    Stores it in state['lifecycle_df'].
    Does NOT merge yet — merge happens after /simulate.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
 
    f = request.files["file"]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400
 
    state["lifecycle_df"] = df
    state["matlab_df"]    = None   # reset stale MATLAB data
    state["merged_df"]    = None
 
    # Return first 100 rows for the Dataset table in the UI
    preview = df.head(100)
    records = []
    for _, row in preview.iterrows():
        records.append({
            "time":        str(row.get("time", row.get("day", row.name))),
            "pressure":    round(float(row.get("pressure", 0)), 4) if "pressure" in row else "—",
            "temperature": round(float(row.get("temperature", 0)), 4) if "temperature" in row else "—",
            "fault":       str(row.get("fault", "—")),
        })
 
    return jsonify({
        "message":      f"Uploaded {len(df)} rows (days 1–{len(df)})",
        "total_rows":   len(df),
        "dataset":      records,
    })
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ROUTE — POST /simulate
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/simulate", methods=["POST"])
def simulate():
    """
    1. Runs MATLAB for the selected faults (produces ONE day of sensor data).
    2. Appends that MATLAB row to the uploaded lifecycle CSV.
    3. Saves the merged DataFrame in state['merged_df'].
    4. Trains models on the merged data.
    5. Generates the RUL graph.
    """
    body   = request.get_json(force=True)
    faults = body.get("faults", [])
 
    if not faults:
        return jsonify({"error": "No faults selected"}), 400
 
    if state["lifecycle_df"] is None:
        return jsonify({
            "error": "No lifecycle CSV uploaded yet. "
                     "Upload your CSV first (Steps 1-3), then run the simulation."
        }), 400
 
    # ── 1. Run MATLAB ──────────────────────────────────────────────────────
    try:
        matlab_df = run_matlab(faults)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"MATLAB failed: {str(e)}"}), 500
 
    state["matlab_df"] = matlab_df
 
    # ── 2. Align MATLAB columns with lifecycle CSV and append ──────────────
    lifecycle_df = state["lifecycle_df"].copy()
 
    # Keep only columns that exist in BOTH dataframes so concat is clean.
    # Any column in matlab_df but not lifecycle_df is dropped (or vice-versa).
    common_cols = [c for c in lifecycle_df.columns if c in matlab_df.columns]
 
    if not common_cols:
        return jsonify({
            "error": "MATLAB output columns do not match the uploaded CSV columns. "
                     "Ensure your MATLAB script writes the same column headers."
        }), 500
 
    matlab_aligned = matlab_df[common_cols].copy()
    lifecycle_aligned = lifecycle_df[common_cols].copy()
 
    # Tag the MATLAB row with the next day number if a day/time column exists
    day_col = next((c for c in ["day", "time", "timestamp"] if c in common_cols), None)
    if day_col:
        last_day = lifecycle_aligned[day_col].max()
        matlab_aligned[day_col] = last_day + 1
 
    merged_df = pd.concat([lifecycle_aligned, matlab_aligned], ignore_index=True)
    state["merged_df"] = merged_df
 
    # ── 3. Train models on merged data ─────────────────────────────────────
    try:
        clf, reg, le, feature_cols = train_models(merged_df)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Model training failed: {str(e)}"}), 500
 
    # ── 4. Generate RUL graph over the full lifecycle ──────────────────────
    try:
        X_all = merged_df[feature_cols].values
        predicted_rul_all = reg.predict(X_all)
        generate_rul_graph(merged_df, predicted_rul_all)
        graph_generated = True
    except Exception as e:
        traceback.print_exc()
        graph_generated = False
 
    # ── 5. Build dataset preview for UI ───────────────────────────────────
    preview = merged_df.tail(20)   # show last 20 rows (includes the new MATLAB day)
    records = []
    for _, row in preview.iterrows():
        records.append({
            "time":        str(row.get("time", row.get("day", row.name))),
            "pressure":    round(float(row.get("pressure", 0)), 4) if "pressure" in merged_df.columns else "—",
            "temperature": round(float(row.get("temperature", 0)), 4) if "temperature" in merged_df.columns else "—",
            "fault":       str(row.get("fault", "—")),
        })
 
    return jsonify({
        "message":         f"MATLAB row appended. Merged lifecycle has {len(merged_df)} days.",
        "total_rows":      len(merged_df),
        "matlab_day":      int(merged_df.shape[0]),   # the day number of the appended row
        "graph_generated": graph_generated,
        "dataset":         records,
    })
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ROUTE — POST /predict
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts fault type and RUL for the LAST row of the merged lifecycle
    (i.e. the MATLAB-appended day), then returns results.
    currentDay from the UI is accepted but the prediction always targets
    the appended MATLAB day (last row).
    """
    if state["merged_df"] is None:
        return jsonify({"error": "No merged dataset. Upload CSV and run simulation first."}), 400
 
    if state["model_clf"] is None:
        return jsonify({"error": "Models not trained yet. Run simulation first."}), 400
 
    body        = request.get_json(force=True)
    current_day = int(body.get("currentDay", len(state["merged_df"])))
 
    merged_df    = state["merged_df"]
    clf          = state["model_clf"]
    reg          = state["model_reg"]
    le           = state["label_enc"]
    feature_cols = state["feature_cols"]
 
    # Predict on the last (MATLAB) row
    last_row = merged_df.iloc[[-1]][feature_cols]
    fault_enc = clf.predict(last_row)[0]
    fault_label = le.inverse_transform([fault_enc])[0]
    rul_pred = float(reg.predict(last_row)[0])
 
    # Predict on ALL rows for graph
    X_all = merged_df[feature_cols].values
    predicted_rul_all = reg.predict(X_all)
    generate_rul_graph(merged_df, predicted_rul_all)
 
    return jsonify({
        "fault":       fault_label,
        "rul":         round(rul_pred, 2),
        "current_day": current_day,
        "matlab_day":  int(merged_df.shape[0]),
        "total_days":  int(merged_df.shape[0]),
    })
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ROUTE — GET /validation_graph
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/validation_graph", methods=["GET"])
def validation_graph():
    """
    Returns a PNG comparing actual vs predicted RUL up to currentDay.
    The MATLAB-appended last day is highlighted.
    """
    if state["merged_df"] is None or state["model_reg"] is None:
        return jsonify({"error": "No data. Run simulation and prediction first."}), 400
 
    current_day = int(request.args.get("currentDay", len(state["merged_df"])))
    merged_df    = state["merged_df"]
    reg          = state["model_reg"]
    feature_cols = state["feature_cols"]
 
    # Slice up to currentDay
    df_slice = merged_df.iloc[:current_day].copy()
    actual_rul   = df_slice["rul"].values if "rul" in df_slice.columns else np.zeros(len(df_slice))
    predicted_rul = reg.predict(df_slice[feature_cols].values)
 
    days = np.arange(1, len(df_slice) + 1)
    total_days = len(merged_df)
    matlab_day = total_days  # MATLAB day is always the last one
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, actual_rul,    color="#22d3ee", linewidth=2, label="Actual RUL")
    ax.plot(days, predicted_rul, color="#f59e0b", linewidth=2,
            linestyle="--", label="Predicted RUL")
 
    if matlab_day <= current_day:
        ax.axvline(x=matlab_day, color="#f43f5e", linestyle=":", linewidth=1.5,
                   label=f"MATLAB Day ({matlab_day})")
        ax.scatter([matlab_day], [predicted_rul[-1]], color="#f43f5e", zorder=5, s=60)
 
    ax.set_xlabel("Operating Day")
    ax.set_ylabel("RUL (days)")
    ax.set_title(f"Validation — Actual vs Predicted RUL (up to Day {current_day})")
    ax.legend()
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
 
    fig.tight_layout()
 
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")
 
 
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=5000)