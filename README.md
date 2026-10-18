# Explainable AI for Predictive Maintenance

Predictive maintenance for a **triplex reciprocating pump**, simulated in MATLAB/Simulink. The system detects which fault(s) a machine has, estimates its **Remaining Useful Life (RUL)**, explains the prediction with **SHAP**, and recommends maintenance steps. It all runs from a React dashboard backed by a Flask API.

> 2nd Semester Experiential Learning (EL) project.

---

## What it does

| Capability | How |
|---|---|
| **Fault classification** | Random Forest classifier over 8 fault configurations: healthy, leak, blockage, bearing wear, and every combination. A majority vote across the lifecycle gives the final class. |
| **Severity estimation** | Multi-output Random Forest regressor predicts the severity of all three faults for each operating day. |
| **RUL prediction** | Fits a power-law degradation curve `s(d) = (d / L)^α` to the predicted severity up to the current day and extrapolates to end-of-life. With multiple faults, the worst case (shortest life) wins. |
| **Validation** | Plots predicted RUL against the true lifecycle and reports the error. |
| **Explainability** | SHAP `TreeExplainer` on the regressor, filtered by degradation direction, shows which signal features drove each fault's severity. |
| **Maintenance suggestions** | Fault-specific descriptions and rectification steps for each detected fault. |
| **PDF report** | One-click export of the prediction, RUL charts, SHAP features and maintenance suggestions. |
| **Live simulation** | The dashboard can launch the Simulink pump model for a chosen fault combination (requires MATLAB). |

## Results

Evaluated on a held-out 20% split of the **4,265-sample** simulated dataset (8 balanced classes):

**Classifier: 85.7% accuracy**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 0.77 | 0.98 | 0.86 |
| Leak | 0.97 | 0.87 | 0.91 |
| Blocking | 0.94 | 0.90 | 0.92 |
| Bearing | 0.84 | 0.82 | 0.83 |
| Leak + Block | 0.98 | 0.95 | 0.97 |
| Block + Bearing | 0.87 | 0.82 | 0.84 |
| Bearing + Leak | 0.73 | 0.77 | 0.75 |
| All faults | 0.80 | 0.72 | 0.76 |

**Severity regressor: overall R² = 0.968** (Leak 0.980 · Blocking 0.988 · Bearing 0.938)

Full metrics are in [`phase_3/model_metrics.json`](phase_3/model_metrics.json).

## Architecture

```text
 ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │  Simulink pump model │ ──▶ │  Feature extraction  │ ──▶ │  Random Forest       │
 │  (fault injection    │     │  14 frequency + flow │     │  classifier +        │
 │   over a lifecycle)  │     │  signal features     │     │  severity regressor  │
 └──────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                      │
 ┌──────────────────────┐     ┌──────────────────────┐                ▼
 │  React dashboard     │ ◀─▶ │  Flask API           │ ◀── RUL curve fit, SHAP,
 │  (Vite)              │     │  new/backend/app.py  │     plots, PDF report
 └──────────────────────┘     └──────────────────────┘
```

**Features (14):** peak frequency, low/mid/high-band spectral power and spectral kurtosis from the pressure signal, plus mean, variance, skewness, kurtosis, peak-to-peak, crest factor, RMS, mean absolute deviation and cumulative-sum range from the flow signal.

## Repository layout

```
.
├── new/                         # Current application
│   ├── backend/
│   │   ├── app.py               # Flask API: upload, simulate, predict, explain, export
│   │   ├── requirements.txt
│   │   └── matlab/              # Simulink pump model + per-fault-class simulation scripts (run_0.m … run_7.m)
│   ├── frontend/                # React + Vite dashboard
│   └── lifecycle_class*.csv     # Sample machine lifecycles, one per fault class, for uploading
├── phase_3/                     # Final models
│   ├── train_models.py          # Trains classifier, regressor and scaler
│   ├── dataset_final.csv        # 4,265-sample training dataset
│   ├── *.pkl                    # Trained models loaded by the backend
│   └── model_metrics.json
├── build_siminputs_multi_fault.m   # Dataset generation step 1: build Simulink inputs for all 8 classes
├── run_siminput_chunk.m            # Step 2: run one chunk of simulations (split across machines)
├── extract_features.m              # Step 3: turn raw simulation logs into the feature dataset
├── paper_content.md                # Methodology and results write-up
└── old/                            # Earlier Streamlit prototype (4 fault classes)
```

## Getting started

### Prerequisites

- Python 3 (developed on 3.14)
- Node.js 20+ and npm
- *Optional:* MATLAB R2025b with Simulink, Simscape and the Predictive Maintenance Toolbox, needed only for the live **Simulate** button and for regenerating the dataset

### 1. Backend

```bash
cd new/backend
pip install -r requirements.txt
python app.py
```

The API runs at `http://127.0.0.1:5000`. Models are loaded from `phase_3/` at startup.

> The MATLAB executable path is hard-coded in `run_matlab()` in `app.py` (`C:\Program Files\MATLAB\R2025b\bin\matlab.exe`). Change it to match your install if you want to use simulation.

### 2. Frontend

```bash
cd new/frontend
npm install
npm run dev
```

Open the URL that Vite prints (usually `http://localhost:5173`).

### 3. Try it

1. *(Optional)* Pick one or more faults and click **Simulate** to run the Simulink model.
2. Upload a lifecycle CSV, for example `new/lifecycle_class4_leak_block.csv`.
3. Choose the current operating day and run the prediction to get the fault class, RUL and degradation chart.
4. Open **Validation** to compare predicted and actual RUL.
5. Open **Explanation** for the SHAP feature contributions.
6. Open **Suggestions** for maintenance steps.
7. Click **Export Report** to download a PDF summary.

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload a lifecycle CSV (multipart `file`) |
| `POST` | `/simulate` | Run the MATLAB simulation for `{"faults": [...]}` |
| `POST` | `/predict` | Classify, estimate severity and fit RUL up to `{"currentDay": N}` |
| `GET` | `/validation_graph` | Predicted vs. actual RUL plot |
| `POST` | `/explain` | SHAP explanation for the latest prediction |
| `GET` | `/export` | PDF report |

## Regenerating the dataset and models

1. In MATLAB, run `build_siminputs_multi_fault.m`. It builds the `Simulink.SimulationInput` array for every class and lifecycle day, then splits it into chunks.
2. Run `run_siminput_chunk.m` on each chunk. Chunks can be shared across machines, and each one resumes safely if interrupted.
3. Run `extract_features.m` to build the feature and label CSV.
4. Train the models:
   ```bash
   cd phase_3
   python train_models.py
   ```

## Acknowledgements

The pump model (`pdmRecipPump`) and the base feature-extraction approach come from the MathWorks example [*Multi-Class Fault Detection Using Simulated Data*](https://www.mathworks.com/help/predmaint/ug/multi-class-fault-detection-using-simulated-data.html). This project extends it with lifecycle-based fault progression, compound faults, RUL estimation and explainability.

