from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import subprocess
from XAI_in_predictive_maintenance.new.backend.ml_pipeline import *
import pandas as pd
from flask import send_file
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt



app = Flask(__name__)
CORS(app)


@app.route('/')
def home():

    return "Backend Running"



@app.route('/simulate', methods=['POST'])
def simulate():

    print("Starting MATLAB...")

    data = request.json

    selected_faults = data['faults']

    current_day = data['currentDay']

    print(selected_faults)
    print(current_day)

    subprocess.run(
        [
            "matlab",
            "-batch",
            "run('C:/project/backend/matlab/simulate.m')"
        ]
    )

    print("MATLAB Finished")

    df = pd.read_csv(
        "C:/project/backend/matlab/simulation_output.csv"
    )

    # DAYS
    days = list(range(1, 21))

    # FAULT CURVES
    fault_curves = {

        "BearingFault": [
            2,4,7,10,14,18,24,30,38,45,
            53,60,68,74,80,86,90,94,97,100
        ],

        "BlockingFault": [
            100,97,94,90,85,80,74,68,60,53,
            45,38,30,24,18,14,10,7,4,2
        ],

        "LeakFault": [
            5,7,9,12,15,19,24,29,35,42,
            50,58,65,72,78,84,89,93,96,100
        ],

        "CavitationFault": [
            3,6,10,15,21,28,36,45,55,65,
            73,80,86,90,93,95,97,98,99,100
        ]
    }

    # FIGURE
    fig, ax = plt.subplots(
        figsize=(11,5),
        facecolor='#0f1117'
    )

    ax.set_facecolor('#0f1117')

    for spine in ax.spines.values():
        spine.set_color('#2e3250')

    ax.tick_params(colors='#8b92b8')

    ax.xaxis.label.set_color('#8b92b8')
    ax.yaxis.label.set_color('#8b92b8')

    ax.grid(
        True,
        linestyle='--',
        alpha=0.25,
        color='#2e3250'
    )

    # STORE RUL VALUES
    rul_values = []

    # PLOT ALL FAULTS
    for fault in selected_faults:

        if fault not in fault_curves:
            continue

        severity = fault_curves[fault]

        past_days = days[:current_day]
        future_days = days[current_day-1:]

        past_severity = severity[:current_day]
        future_severity = severity[current_day-1:]

        # OBSERVED
        ax.plot(
            past_days,
            past_severity,
            linewidth=3,
            label=f"{fault} Observed"
        )

        # PREDICTED
        ax.plot(
            future_days,
            future_severity,
            linestyle='--',
            linewidth=3,
            label=f"{fault} Predicted"
        )

        # SIMPLE RUL ESTIMATION
        rul = 20 - current_day

        rul_values.append(rul)

    # MINIMUM RUL
    minimum_rul = min(rul_values)

    # CURRENT DAY LINE
    ax.axvline(
        x=current_day,
        color='#ce93d8',
        linestyle=':',
        linewidth=2,
        label=f'Current Day ({current_day})'
    )

    # FAILURE THRESHOLD
    ax.axhline(
        y=100,
        color='#ef5350',
        linestyle='--',
        linewidth=2,
        label='Failure Threshold'
    )

    # TITLE
    ax.set_title(
        f'Triplex Pump RUL Prediction | Estimated RUL: {minimum_rul} days',
        color='#e8eaf0',
        fontsize=16,
        pad=15
    )

    ax.set_xlabel(
        'Operating Days',
        fontsize=12
    )

    ax.set_ylabel(
        'Fault Severity',
        fontsize=12
    )

    # LEGEND
    legend = ax.legend(
        loc='upper left',
        facecolor='#1c1f2e',
        edgecolor='#2e3250'
    )

    for text in legend.get_texts():
        text.set_color('#e8eaf0')

    plt.tight_layout()

    plt.savefig(
        "C:/project/backend/matlab/graph.png",
        dpi=150,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )

    plt.close()

    response_data = {
        "dataset": df.to_dict(orient='records'),
        "minimum_rul": minimum_rul
    }

    return jsonify(response_data)


@app.route('/graph')
def graph():

    return send_file(
        "C:/project/backend/matlab/graph.png",
        mimetype='image/png'
    )

@app.route('/predict', methods=['POST'])
def predict():

    print("Prediction Started")

    return jsonify({

        "fault": "Bearing Fault",

        "rul": 142,

        "current_day": 20

    })
    
    
if __name__ == "__main__":

    app.run(debug=True)