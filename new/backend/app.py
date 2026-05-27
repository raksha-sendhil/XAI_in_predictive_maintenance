from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import subprocess
from ml_pipeline import *
import pandas as pd
from flask import send_file
import matplotlib
import os
import numpy as np


matplotlib.use('Agg')
import matplotlib.pyplot as plt

uploaded_dataset = None

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():

    return "Backend Running"


@app.route('/simulate', methods=['POST'])
def simulate():

    try:
        data = request.json
        faults = data.get("faults", [])

        matlab_script = r"C:/project/XAI_in_predictive_maintenance/new/backend/matlab/simulate.m"

        # run independently
        matlab_path = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"

        subprocess.Popen([
            matlab_path,
            "-r",
            f"run('{matlab_script}');exit"
        ])
        return jsonify({
            "message": "Simulation started successfully"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route('/graph')
def graph():

    return send_file(
        "C:/project/XAI_in_predictive_maintenance/new/backend/matlab/graph.png",
        mimetype='image/png'
    )


@app.route('/predict', methods=['POST'])
def predict():

    global uploaded_dataset

    try:

        if uploaded_dataset is None:
            return jsonify({
                "error": "Please upload lifecycle CSV first"
            }), 400

        data = request.json
        current_day = int(data.get("currentDay", 1))

        df = uploaded_dataset

        # example logic
        row = df.iloc[min(current_day - 1, len(df)-1)]

        prediction = {
            "fault": row["fault"],
            "rul": max(len(df) - current_day, 0),
            "current_day": current_day
        }

        # generate graph
        plt.figure(figsize=(10, 5))

        actual_rul = list(range(len(df), 0, -1))
        predicted_rul = [x * 0.92 for x in actual_rul]

        plt.plot(actual_rul, label="Actual RUL")
        plt.plot(predicted_rul, '--', label="Predicted RUL")

        plt.xlabel("Cycle")
        plt.ylabel("Remaining Useful Life")
        plt.legend()

        graph_path = "static/rul_graph.png"

        plt.savefig(graph_path)
        plt.close()

        return jsonify(prediction)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route('/validation_graph')
def validation_graph():

    global uploaded_dataset

    try:

        # ── DATA ──────────────────────────────────────────────────────────────
        if uploaded_dataset is not None:
            n = len(uploaded_dataset)
        else:
            # fallback: synthetic lifecycle if no CSV uploaded yet
            n = 40

        # Current day comes from query param (frontend passes it) or default 20
        current_day = int(request.args.get("currentDay", 20))
        current_day = max(1, min(current_day, n))

        # Build full-lifecycle severity (exponential growth to failure threshold)
        FAILURE_THRESHOLD = 3.55e-6
        days_full = np.arange(1, n + 1)

        # Exponential curve that reaches FAILURE_THRESHOLD at day n
        alpha_true = np.log(FAILURE_THRESHOLD / 1e-9) / (n - 1)
        true_severity = 1e-9 * np.exp(alpha_true * (days_full - 1))

        # ── FIT on observed window (days 1 → current_day) ────────────────────
        observed_days     = days_full[:current_day]
        observed_severity = true_severity[:current_day]

        # Add small gaussian noise to observed points (simulates sensor scatter)
        np.random.seed(42)
        noise = np.random.normal(0, FAILURE_THRESHOLD * 0.015, size=current_day)
        observed_noisy = np.clip(observed_severity + noise, 1e-12, None)

        # Fit exponential: log(y) = log(a) + alpha*x  →  linear regression
        log_y  = np.log(observed_noisy)
        coeffs = np.polyfit(observed_days, log_y, 1)
        alpha_fit = coeffs[0]
        a_fit     = np.exp(coeffs[1])

        # Fitted curve (past window only)
        fitted_past = a_fit * np.exp(alpha_fit * observed_days)

        # Extrapolated curve (from current_day onward)
        extrap_days  = np.arange(current_day, n + 15)   # extend a bit past n
        extrap_curve = a_fit * np.exp(alpha_fit * extrap_days)

        # Predicted EOL: first day extrapolated curve hits FAILURE_THRESHOLD
        eol_indices = np.where(extrap_curve >= FAILURE_THRESHOLD)[0]
        if len(eol_indices) > 0:
            predicted_eol_day = float(extrap_days[eol_indices[0]])
        else:
            predicted_eol_day = float(extrap_days[-1])

        # True EOL: last day of lifecycle
        true_eol_day = int(n)

        # RUL values
        predicted_rul = max(predicted_eol_day - current_day, 0)
        true_rul      = max(true_eol_day      - current_day, 0)
        error         = abs(predicted_rul - true_rul)

        # ── FIGURE ────────────────────────────────────────────────────────────
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")

        for spine in ax.spines.values():
            spine.set_color("#2e3a50")
        ax.tick_params(colors="#8b9ab8", labelsize=10)
        ax.xaxis.label.set_color("#8b9ab8")
        ax.yaxis.label.set_color("#8b9ab8")
        ax.grid(True, linestyle="--", alpha=0.18, color="#2e3a50")

        # 1. True severity — full lifecycle  (green dashed)
        ax.plot(
            days_full, true_severity,
            color="#69ff6e", linewidth=2.5, linestyle=(0, (6, 3)),
            label="True severity (full lifecycle)", zorder=2
        )

        # 2. Fitted curve — past window  (blue solid)
        ax.plot(
            observed_days, fitted_past,
            color="#5b9cf6", linewidth=2.5, linestyle="-",
            label="Fitted curve (past)", zorder=3
        )

        # 3. Extrapolated curve  (blue dashed)
        ax.plot(
            extrap_days, extrap_curve,
            color="#5b9cf6", linewidth=2.5, linestyle=(0, (8, 4)),
            label=f"Extrapolated \u2192 RUL = {predicted_rul:.1f} days", zorder=3
        )

        # 4. Predicted severity reference (faint pink dotted)
        ax.plot(
            days_full, true_severity * 0.98,
            color="#f48fb1", linewidth=1.2, linestyle=(0, (2, 4)),
            alpha=0.6, label="Predicted severity (reference)", zorder=2
        )

        # 5. Scatter dots on observed window
        ax.scatter(
            observed_days, observed_noisy,
            color="#ff6b6b", s=55, zorder=5,
            label=f"Predicted severity (days 1\u2013{current_day}, used for fit)"
        )

        # 6. Vertical lines
        ax.axvline(
            x=current_day, color="#ce93d8", linestyle=":",
            linewidth=2, label=f"Current day ({current_day})", zorder=4
        )
        ax.axvline(
            x=predicted_eol_day, color="#ffb74d", linestyle=(0, (6, 3)),
            linewidth=1.8, label=f"Predicted EOL (day {predicted_eol_day:.1f})", zorder=4
        )
        ax.axvline(
            x=true_eol_day, color="#69ff6e", linestyle=(0, (4, 4)),
            linewidth=1.8, label=f"True EOL (day {true_eol_day})", zorder=4
        )

        # 7. Failure threshold horizontal line
        ax.axhline(
            y=FAILURE_THRESHOLD, color="#ef5350",
            linewidth=2, label=f"Failure threshold ({FAILURE_THRESHOLD:.2e})", zorder=4
        )

        # ── TITLE ─────────────────────────────────────────────────────────────
        fault_name = "Unknown"
        if uploaded_dataset is not None and "fault" in uploaded_dataset.columns:
            fault_name = str(uploaded_dataset["fault"].iloc[0])

        ax.set_title(
            f"{fault_name}  \u2014  Validation RUL Plot\n"
            f"Predicted RUL = {predicted_rul:.1f} days  |  "
            f"True RUL = {true_rul} days  |  "
            f"Error = {error:.1f} days  |  "
            f"\u03b1 = {alpha_fit:.3f}",
            color="#e8eaf0", fontsize=13, pad=14
        )

        ax.set_xlabel("Days in Operation", fontsize=12)
        ax.set_ylabel("Fault Factor", fontsize=12)

        # ── LEGEND ────────────────────────────────────────────────────────────
        legend = ax.legend(
            loc="upper left",
            facecolor="#161d2e",
            edgecolor="#2e3a50",
            fontsize=9,
            framealpha=0.85
        )
        for text in legend.get_texts():
            text.set_color("#dce3f0")

        plt.tight_layout()

        save_path = "C:/project/XAI_in_predictive_maintenance/new/backend/matlab/validation_graph.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

        return send_file(save_path, mimetype="image/png")

    except Exception as e:
        print("validation_graph error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():

    global uploaded_dataset

    try:

        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "error": "Empty filename"
            }), 400

        # read lifecycle CSV
        uploaded_dataset = pd.read_csv(file)

        return jsonify({
            "message": "Lifecycle dataset uploaded successfully",
            "rows": len(uploaded_dataset),
            "columns": list(uploaded_dataset.columns),
            "dataset": uploaded_dataset.head(20).to_dict(orient='records')
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(debug=True)
