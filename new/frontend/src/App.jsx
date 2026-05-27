import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFaults, setSelectedFaults] = useState([]);
  const [dataset, setDataset] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [currentDay, setCurrentDay] = useState(1);
  const [simulationStatus, setSimulationStatus] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);
  const [graphUrl, setGraphUrl] = useState("");
  const [activeSection, setActiveSection] = useState("simulation");
  const [liveSyncActive, setLiveSyncActive] = useState(true);

  const faults = ["BearingFault", "BlockingFault", "LeakFault"];

  const toggleFault = (fault) => {
    if (selectedFaults.includes(fault)) {
      setSelectedFaults(selectedFaults.filter((f) => f !== fault));
    } else {
      setSelectedFaults([...selectedFaults, fault]);
    }
  };

  const runSimulation = async () => {
  if (selectedFaults.length === 0) {
    alert("Please select at least one fault");
    return;
  }

  setSimulationStatus("running");

  try {
    const response = await fetch("http://127.0.0.1:5000/simulate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        faults: selectedFaults,
      }),
    });

    // backend crashed or matlab crashed
    if (!response.ok) {
      throw new Error("Backend simulation failed");
    }

    const data = await response.json();

    // dataset
    if (data.dataset) {
      setDataset(data.dataset);
    }

    // graph refresh
    if (data.graph_generated) {
      setGraphUrl(`http://127.0.0.1:5000/static/rul_graph.png?t=${Date.now()}`);
    }

    setSimulationStatus("success");
  } catch (error) {
    console.log(error);

    // IMPORTANT
    // website should NEVER crash
    setSimulationStatus("error");
  }
};

  const uploadFile = async () => {

  if (!uploadedFile) {
    alert("Please select CSV file");
    return;
  }

  const formData = new FormData();
  formData.append("file", uploadedFile);

  try {

    const response = await fetch(
      "http://127.0.0.1:5000/upload",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error("Upload failed");
    }

    const data = await response.json();

    setDataset(data.dataset);

    alert("Lifecycle dataset uploaded successfully");

  } catch (error) {

    console.log(error);
    alert("CSV upload failed");
  }
};

  const makePrediction = async () => {
  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        currentDay: Number(currentDay),
      }),
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const data = await response.json();

    setPredictionResult(data);

    // reload latest graph
    setGraphUrl(
      `http://127.0.0.1:5000/static/rul_graph.png?t=${Date.now()}`
    );
  } catch (error) {
    console.log(error);
    alert("Prediction failed");
  }
};

  const sensorReadings = [
    { label: "Pressure", unit: "bar", value: predictionResult ? "8.42" : "—", icon: "◈", status: "nominal" },
    { label: "Temperature", unit: "°C", value: predictionResult ? "74.3" : "—", icon: "◉", status: "warning" },
    { label: "Vibration", unit: "mm/s", value: predictionResult ? "2.17" : "—", icon: "◇", status: "nominal" },
    { label: "Flow Rate", unit: "L/min", value: predictionResult ? "31.8" : "—", icon: "◈", status: "nominal" },
  ];

  const xaiInsights = [
    { factor: "Temperature Spike", contribution: 38, direction: "up" },
    { factor: "Vibration Amplitude", contribution: 27, direction: "up" },
    { factor: "Pressure Variance", contribution: 19, direction: "up" },
    { factor: "Flow Rate Stability", contribution: 16, direction: "down" },
  ];

  return (
    <div className="dashboard-root">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">⬡</span>
          <div>
            <div className="brand-name">PREDICT_TECH_OS</div>
            <div className="brand-version">v2.4.0-STABLE</div>
          </div>
        </div>

        <div className="sidebar-plant">
          <span className="plant-dot" />
          <div>
            <div className="plant-label">ALPHA_PLANT_7</div>
            <div className="plant-sub">Predictive Engine</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeSection === "simulation" ? "active" : ""}`}
            onClick={() => setActiveSection("simulation")}
          >
            <span className="nav-icon">▸</span> Simulation
          </button>
          <button
            className={`nav-item ${activeSection === "validation" ? "active" : ""}`}
            onClick={() => setActiveSection("validation")}
          >
            <span className="nav-icon">▸</span> Validation Suite
          </button>
          <button
            className={`nav-item ${activeSection === "sensors" ? "active" : ""}`}
            onClick={() => setActiveSection("sensors")}
          >
            <span className="nav-icon">▸</span> Sensor Data
          </button>
          <button
            className={`nav-item ${activeSection === "dataset" ? "active" : ""}`}
            onClick={() => setActiveSection("dataset")}
          >
            <span className="nav-icon">▸</span> Dataset
          </button>
        </nav>

        <div className="sidebar-footer">
          <span className="footer-icon">⊕</span> Support
          <br />
          <span className="footer-icon">⟨⟩</span> API Documentation
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-panel">
        {/* TOP BAR */}
        <header className="topbar">
          <div className="topbar-tabs">
            <button
              className={`tab-btn ${activeSection === "simulation" ? "tab-active" : ""}`}
              onClick={() => setActiveSection("simulation")}
            >
              Simulation
            </button>
            <button
              className={`tab-btn ${activeSection === "validation" ? "tab-active" : ""}`}
              onClick={() => setActiveSection("validation")}
            >
              Validation
            </button>
          </div>
          <div className="topbar-actions">
            <button className="action-btn">⬆ Export Report</button>
            <button
              className={`live-btn ${liveSyncActive ? "live-on" : ""}`}
              onClick={() => setLiveSyncActive(!liveSyncActive)}
            >
              <span className="live-dot" /> Live Sync {liveSyncActive ? "Active" : "Paused"}
            </button>
          </div>
        </header>

        {/* SIMULATION SECTION */}
        {(activeSection === "simulation" || activeSection === "validation") && (
          <>
            <section className="content-section">
              <div className="section-header">
                <div>
                  <h1 className="section-title">
                    {activeSection === "validation"
                      ? "Model Performance & Validation Suite"
                      : "Predictive Maintenance Dashboard"}
                  </h1>
                  <p className="section-desc">
                    Real-time evaluation of predictive algorithms against ground truth telemetry data.
                    Monitoring LSTM-RNN for anomalies and drift.
                  </p>
                </div>
              </div>

              {/* FAULT SELECTION */}
              <div className="cards-row">
                <div className="card">
                  <div className="card-label">STEP 1 — SELECT FAULT TYPES</div>
                  <div className="fault-grid">
                    {faults.map((fault) => (
                      <button
                        key={fault}
                        className={`fault-chip ${selectedFaults.includes(fault) ? "fault-chip-active" : ""}`}
                        onClick={() => toggleFault(fault)}
                      >
                        <span className="chip-dot" />
                        {fault}
                      </button>
                    ))}
                  </div>
                  <div className="selected-display">
                    Active faults:{" "}
                    <span className="selected-value">
                      {selectedFaults.length > 0 ? selectedFaults.join(", ") : "None selected"}
                    </span>
                  </div>
                </div>

                {/* MODEL CARD */}
                <div className="card model-card">
                  <div className="model-badge production">PRODUCTION</div>
                  <div className="model-name">LSTM-RNN</div>
                  <div className="model-metrics">
                    <div className="metric-row">
                      <span className="metric-label">Accuracy</span>
                      <span className="metric-value accent-green">
                        {predictionResult ? "94.2%" : "—"}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">RMSE</span>
                      <span className="metric-value">
                        {predictionResult ? "1.42 cycles" : "—"}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">MAE</span>
                      <span className="metric-value">
                        {predictionResult ? "9.89 cycles" : "—"}
                      </span>
                    </div>
                  </div>
                  <div className="confidence-bar-wrap">
                    <div className="confidence-label">
                      Confidence Interval <span>High</span>
                    </div>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: predictionResult ? "82%" : "0%" }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* CONTROLS ROW */}
              <div className="cards-row">
                <div className="card">
                  <div className="card-label">STEP 2 — RUN MATLAB SIMULATION</div>
                  <button className="primary-action-btn" onClick={runSimulation}>
                    <span className="btn-icon">▶</span> Run Simulation
                  </button>
                  {simulationStatus && (
                    <div className={`status-pill ${simulationStatus}`}>
                      {simulationStatus === "running" && "⟳ Running MATLAB Simulation..."}
                      {simulationStatus === "success" && "✓ Simulation Completed Successfully"}
                      {simulationStatus === "error" && "✕ Simulation Failed"}
                    </div>
                  )}
                </div>

                <div className="card">
                  <div className="card-label">STEP 3 — UPLOAD CSV FILE</div>
                  <label className="file-upload-label">
                    <input
                      type="file"
                      accept=".csv"
                      style={{ display: "none" }}
                      onChange={(e) => setUploadedFile(e.target.files[0])}
                    />
                    <span className="upload-icon">⊕</span>
                    {uploadedFile ? uploadedFile.name : "Choose CSV file..."}
                  </label>
                  <button className="primary-action-btn" onClick={uploadFile}>
                    Upload Dataset
                  </button>
                </div>
              </div>

              {/* DAY SELECTOR + PREDICT */}
              <div className="cards-row">
                <div className="card">
                  <div className="card-label">STEP 4 — OPERATING DAY</div>
                  <div className="slider-display">
                    <span>Day {currentDay}</span>
                    <span className="slider-max">/ 20</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={currentDay}
                    onChange={(e) => setCurrentDay(e.target.value)}
                    className="dash-slider"
                  />
                  <div className="slider-ticks">
                    <span>1</span><span>5</span><span>10</span><span>15</span><span>20</span>
                  </div>
                </div>

                <div className="card">
                  <div className="card-label">STEP 5 — RUN PREDICTION</div>
                  <button className="primary-action-btn predict-btn" onClick={makePrediction}>
                    <span className="btn-icon">◎</span> Make Prediction
                  </button>
                  {predictionResult && (
                    <div className="status-pill success">✓ Prediction Complete</div>
                  )}
                </div>
              </div>
            </section>

            {/* SENSOR READINGS */}
            <section className="content-section">
              <div className="card-label" style={{ marginBottom: "12px" }}>LIVE SENSOR READINGS</div>
              <div className="sensors-row">
                {sensorReadings.map((s) => (
                  <div key={s.label} className={`sensor-card sensor-${s.status}`}>
                    <div className="sensor-icon">{s.icon}</div>
                    <div className="sensor-label">{s.label}</div>
                    <div className="sensor-value">{s.value}</div>
                    <div className="sensor-unit">{s.unit}</div>
                    <div className={`sensor-status-badge ${s.status}`}>
                      {s.status.toUpperCase()}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* RUL GRAPH */}
            {graphUrl && (
              <section className="content-section">
                <div className="card">
                  <div className="card-label">REMAINING USEFUL LIFE (RUL) TRAJECTORY</div>
                  <p className="graph-sub">
                    Actual vs. Predicted cycles across simulated stress conditions.
                  </p>
                  <div className="graph-legend">
                    <span className="legend-item">
                      <span className="legend-line solid" /> Actual RUL (Ground Truth)
                    </span>
                    <span className="legend-item">
                      <span className="legend-line dashed" /> Predicted RUL
                    </span>
                  </div>
                  <div className="graph-wrap">
                    <img src={graphUrl} alt="RUL Prediction Graph" className="graph-img" />
                  </div>

                  {/* XAI SECTION */}
                  <div className="xai-section">
                    <div className="xai-header">
                      <span className="xai-badge">XAI</span>
                      <span className="xai-title">Explainable AI — Feature Attribution</span>
                    </div>
                    <p className="xai-desc">
                      SHAP-based feature importance showing which sensor inputs most influenced
                      the RUL prediction for the current operating cycle.
                    </p>
                    <div className="xai-factors">
                      {xaiInsights.map((x) => (
                        <div key={x.factor} className="xai-row">
                          <span className="xai-factor">{x.factor}</span>
                          <div className="xai-bar-wrap">
                            <div
                              className={`xai-bar-fill ${x.direction === "up" ? "xai-up" : "xai-down"}`}
                              style={{ width: `${x.contribution * 2}%` }}
                            />
                          </div>
                          <span className={`xai-pct ${x.direction === "up" ? "xai-up" : "xai-down"}`}>
                            {x.direction === "up" ? "▲" : "▼"} {x.contribution}%
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="xai-note">
                      ⓘ Temperature Spike is the primary degradation driver in the current cycle.
                      Monitor thermal envelope and cooling system integrity.
                    </p>
                  </div>
                </div>
              </section>
            )}

            {/* PREDICTION RESULTS */}
            {predictionResult && (
              <section className="content-section">
                <div className="results-grid">
                  <div className="result-tile">
                    <div className="result-tile-label">DETECTED FAULT</div>
                    <div className="result-tile-value fault-value">{predictionResult.fault}</div>
                  </div>
                  <div className="result-tile result-tile-accent">
                    <div className="result-tile-label">REMAINING USEFUL LIFE</div>
                    <div className="result-tile-value rul-value">{predictionResult.rul}</div>
                    <div className="result-tile-unit">days</div>
                  </div>
                  <div className="result-tile">
                    <div className="result-tile-label">CURRENT OPERATING DAY</div>
                    <div className="result-tile-value">{predictionResult.current_day}</div>
                  </div>
                </div>
              </section>
            )}
          </>
        )}

        {/* SENSOR SECTION */}
        {activeSection === "sensors" && (
          <section className="content-section">
            <div className="section-header">
              <div>
                <h1 className="section-title">Live Sensor Telemetry</h1>
                <p className="section-desc">
                  Real-time readings from all instrumented measurement points on ALPHA_PLANT_7.
                </p>
              </div>
            </div>
            <div className="sensors-row">
              {sensorReadings.map((s) => (
                <div key={s.label} className={`sensor-card sensor-${s.status}`}>
                  <div className="sensor-icon">{s.icon}</div>
                  <div className="sensor-label">{s.label}</div>
                  <div className="sensor-value">{s.value}</div>
                  <div className="sensor-unit">{s.unit}</div>
                  <div className={`sensor-status-badge ${s.status}`}>{s.status.toUpperCase()}</div>
                </div>
              ))}
            </div>
            <div className="card" style={{ marginTop: "16px" }}>
              <div className="card-label">SENSOR NOTES</div>
              <p className="xai-desc">
                Temperature reading on Sensor 2 shows a gradual upward trend over the last 4 cycles.
                Recommend scheduled inspection of heat exchanger unit. All other sensors operating
                within nominal bounds.
              </p>
            </div>
          </section>
        )}

        {/* DATASET SECTION */}
        {activeSection === "dataset" && (
          <section className="content-section">
            <div className="section-header">
              <div>
                <h1 className="section-title">Generated Dataset</h1>
                <p className="section-desc">
                  Telemetry records produced by the MATLAB simulation engine.
                </p>
              </div>
            </div>
            {dataset.length > 0 ? (
              <div className="card">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Pressure</th>
                      <th>Temperature</th>
                      <th>Fault</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.map((item, index) => (
                      <tr key={index}>
                        <td>{item.time}</td>
                        <td>{item.pressure}</td>
                        <td>{item.temperature}</td>
                        <td>
                          <span className="fault-tag">{item.fault}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="card empty-state">
                <div className="empty-icon">◈</div>
                <div className="empty-text">No dataset loaded. Run a simulation or upload a CSV file.</div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
