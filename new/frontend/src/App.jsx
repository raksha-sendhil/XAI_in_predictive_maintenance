import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFaults, setSelectedFaults] = useState([]);
  const [dataset, setDataset] = useState([]);
  const [uploadedDataset, setUploadedDataset] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [currentDay, setCurrentDay] = useState(0);
  const [simulationStatus, setSimulationStatus] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);
  const [graphUrl, setGraphUrl] = useState("");
  const [activeSection, setActiveSection] = useState("simulation");
  const [liveSyncActive, setLiveSyncActive] = useState(true);
  const [validationGraph, setValidationGraph] = useState("");
  const [isSimulating, setIsSimulating] = useState(false); // guard against double-clicks

  const faults = ["BearingFault", "BlockingFault", "LeakFault"];

  const toggleFault = (fault) => {
    setSelectedFaults((prev) =>
      prev.includes(fault) ? prev.filter((f) => f !== fault) : [...prev, fault]
    );
  };

  const runSimulation = async () => {
    if (isSimulating) return; // prevent double-click during MATLAB run

    setIsSimulating(true);
    setSimulationStatus("running");
    // Clear previous results so stale graphs don't linger
    setGraphUrl("");
    setPredictionResult(null);
    setValidationGraph("");

    try {
      const response = await fetch("http://127.0.0.1:5000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ faults: selectedFaults }),
      });

      if (!response.ok) throw new Error("Backend simulation failed");

      setSimulationStatus("success");
    } catch (error) {
      console.error("Simulation error:", error);
      setSimulationStatus("error");
    } finally {
      setIsSimulating(false);
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
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      setDataset(data.dataset);
      setUploadedDataset(data.dataset || []);
      alert("Lifecycle dataset uploaded successfully");
    } catch (error) {
      console.error("Upload error:", error);
      alert("CSV upload failed");
    }
  };

  const makePrediction = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentDay: Number(currentDay) }),
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      const ts = Date.now();

      setPredictionResult(data);

      // Validation graph: backend renders actual-vs-predicted up to currentDay
      setValidationGraph(
        `http://127.0.0.1:5000/validation_graph?currentDay=${Number(currentDay)}&t=${ts}`
      );

      // Refresh RUL graph with new timestamp so browser doesn't serve cached version
      setGraphUrl(`http://127.0.0.1:5000/static/rul_graph.png?t=${ts}`);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Prediction failed");
    }
  };


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
            className={`nav-item ${activeSection === "dataset" ? "active" : ""}`}
            onClick={() => setActiveSection("dataset")}
          >
            <span className="nav-icon">▸</span> Dataset
          </button>
          <button
            className={`nav-item ${activeSection === "explanation" ? "active" : ""}`}
            onClick={() => setActiveSection("explanation")}
          >
            <span className="nav-icon">▸</span> Explanation
          </button>
        </nav>

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

        {/* ─── SIMULATION SECTION ─── */}
        {activeSection === "simulation" && (
          <>
            <section className="content-section">
              <div className="section-header">
                <div>
                  <h1 className="section-title">Predictive Maintenance Dashboard</h1>
                  <p className="section-desc">
                    Configure fault types, run the MATLAB simulation, and generate RUL predictions
                    for ALPHA_PLANT_7.
                  </p>
                </div>
              </div>

              {/* FAULT SELECTION */}
              <div className="card" style={{ marginBottom: "16px" }}>
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
                    {selectedFaults.length > 0 ? selectedFaults.join(", ") : "None (Healthy)"}
                  </span>
                </div>
              </div>

              {/* CONTROLS ROW */}
              <div className="cards-row">
                <div className="card">
                  <div className="card-label">STEP 2 — RUN MATLAB SIMULATION</div>
                  <button
                    className="primary-action-btn"
                    onClick={runSimulation}
                    disabled={isSimulating}
                    style={{ opacity: isSimulating ? 0.6 : 1, cursor: isSimulating ? "not-allowed" : "pointer" }}
                  >
                    <span className="btn-icon">{isSimulating ? "⟳" : "▶"}</span>
                    {isSimulating ? "Running..." : "Run Simulation"}
                  </button>
                  {simulationStatus && (
                    <div className={`status-pill ${simulationStatus}`}>
                      {simulationStatus === "running" && "⟳ Running MATLAB Simulation..."}
                      {simulationStatus === "success" && "✓ Simulation Started Successfully"}
                      {simulationStatus === "error" && "✕ Simulation Failed — check Flask logs"}
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
                    <span className="slider-max">/ 45</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="45"
                    value={currentDay}
                    onChange={(e) => setCurrentDay(e.target.value)}
                    className="dash-slider"
                  />
                  <div className="slider-ticks">
                    <span>0</span><span>10</span><span>20</span><span>30</span><span>45</span>
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

            {/* RUL GRAPH — shown only after prediction */}
            {predictionResult && graphUrl && (
              <section className="content-section">
                <div className="card">
                  <div className="card-label">
                    REMAINING USEFUL LIFE — UP TO DAY {currentDay}
                  </div>
<div className="graph-wrap">
                    <img
                      src={graphUrl}
                      alt="RUL Prediction Graph"
                      className="graph-img"
                      onError={(e) => {
                        e.target.style.display = "none";
                        e.target.nextSibling.style.display = "block";
                      }}
                    />
                    <p style={{ display: "none", color: "var(--color-warning, #f59e0b)", marginTop: 8 }}>
                      ⚠ Graph image could not be loaded. Ensure Flask is serving /static/rul_graph.png
                    </p>
                  </div>
                </div>
              </section>
            )}

          </>
        )}

        {/* ─── VALIDATION SECTION ─── */}
        {activeSection === "validation" && (
          <section className="content-section">
            <div className="section-header">
              <div>
                <h1 className="section-title">Model Validation Suite</h1>
                <p className="section-desc">
                  Model validation plot against ground truth data.
                </p>
              </div>
            </div>
            <div className="card">
              <div className="card-label">ACCURACY & VALIDATION PLOT</div>
              {validationGraph ? (
                <div className="graph-wrap">
                  <img
                    src={validationGraph}
                    alt="Validation Graph"
                    className="graph-img"
                    onError={(e) => {
                      e.target.style.display = "none";
                      e.target.nextSibling.style.display = "block";
                    }}
                  />
                  <p style={{ display: "none", color: "var(--color-warning, #f59e0b)", marginTop: 8 }}>
                    ⚠ Validation graph could not be loaded. Ensure Flask is serving /validation_graph
                  </p>
                </div>
              ) : (
                <div className="empty-state" style={{ padding: "40px 0" }}>
                  <div className="empty-icon">◈</div>
                  <div className="empty-text">
                    No validation graph yet. Run a prediction to generate the plot.
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ─── EXPLANATION / SHAP SECTION ─── */}
        {activeSection === "explanation" && (
          <section className="content-section">
            <div className="section-header">
              <div>
                <h1 className="section-title">Explainable AI — SHAP Analysis</h1>
                <p className="section-desc">
                  Feature attribution output from SHAP. Shows which sensor inputs most influenced
                  the RUL prediction for the current operating cycle.
                </p>
              </div>
            </div>

            <div className="card">
              <div className="card-label">SHAP FEATURE IMPORTANCE</div>
              {predictionResult ? (
                <>
                  <div className="xai-factors" style={{ marginTop: "16px" }}>
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
                  <p className="xai-note" style={{ marginTop: "16px" }}>
                    ⓘ Temperature Spike is the primary degradation driver in the current cycle.
                    Monitor thermal envelope and cooling system integrity.
                  </p>
                </>
              ) : (
                <div className="empty-state" style={{ padding: "40px 0" }}>
                  <div className="empty-icon">◈</div>
                  <div className="empty-text">
                    No SHAP output available. Run a prediction first to generate explanations.
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ─── SENSOR SECTION ─── */}

        {/* ─── DATASET SECTION ─── */}
        {activeSection === "dataset" && (
          <section className="content-section">
            <div className="section-header">
              <div>
                <h1 className="section-title">Uploaded CSV Dataset</h1>
                <p className="section-desc">
                  {uploadedFile
                    ? `${uploadedDataset.length} records from: ${uploadedFile.name}`
                    : "Upload a CSV file from the Simulation section to view its records here."}
                </p>
              </div>
            </div>
            {uploadedDataset.length > 0 ? (
              <div className="card">
                <div style={{ overflowX: "auto" }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(uploadedDataset[0]).map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {uploadedDataset.map((item, index) => (
                        <tr key={index}>
                          {Object.keys(uploadedDataset[0]).map((col) => (
                            <td key={col}>
                              {col.toLowerCase() === "fault" ? (
                                <span className="fault-tag">{item[col]}</span>
                              ) : (
                                item[col]
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="card empty-state">
                <div className="empty-icon">◈</div>
                <div className="empty-text">
                  No dataset loaded. Upload a CSV file from the Simulation section.
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
