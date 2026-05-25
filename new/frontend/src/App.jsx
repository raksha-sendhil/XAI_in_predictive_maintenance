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
  

  const faults = [
    "BearingFault",
    "BlockingFault",
    "LeakFault"
  ];

  const toggleFault = (fault) => {

    if (selectedFaults.includes(fault)) {

      setSelectedFaults(
        selectedFaults.filter((f) => f !== fault)
      );

    }

    else {

      setSelectedFaults([
        ...selectedFaults,
        fault
      ]);

    }

  };

  // RUN MATLAB SIMULATION

  const runSimulation = async () => {

    if (selectedFaults.length === 0) {

      alert("Please select at least one fault");
      return;

    }

    setSimulationStatus("Running MATLAB Simulation...");

    try {

      const response = await fetch(
        "http://127.0.0.1:5000/simulate",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({

            faults: selectedFaults

          })

        }
      );

      const data = await response.json();

      console.log(data);

      setDataset(data.dataset);

      setSimulationStatus(
        "Simulation Completed Successfully"
      );

    }

    catch (error) {

      console.log(error);

      setSimulationStatus("Simulation Failed");

    }

  };

  // UPLOAD CSV

  const uploadFile = async () => {

    if (!uploadedFile) {

      alert("Please select a CSV file");
      return;

    }

    const formData = new FormData();

    formData.append("file", uploadedFile);

    try {

      const response = await fetch(
        "http://127.0.0.1:5000/upload",
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      console.log(data);

      setDataset(data.dataset);

      setSimulationStatus(
        "Dataset Uploaded Successfully"
      );

    }

    catch (error) {

      console.log(error);

    }

  };

  // PREDICTION

  const makePrediction = async () => {

    try {

      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {

          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({

            currentDay: Number(currentDay)

          })

        }
      );

      const data = await response.json();

      console.log(data);

      setPredictionResult(data);

      setRul(data.rul);

      setGraphUrl(
        `http://127.0.0.1:5000/graph?${new Date().getTime()}`
      );

    }

    catch (error) {

      console.log(error);

    }

  };

  return (

  <div className="app-container">

    <h1 className="dashboard-title">
      ⚙️ Predictive Maintenance Dashboard
    </h1>

    <p className="dashboard-sub">
      MATLAB + AI Based Fault Detection & RUL Prediction
    </p>

    {/* STEP 1 */}

    <div className="section">

      <h2>1. Select Fault Types</h2>

      <div className="fault-buttons">

        {

          faults.map((fault) => (

            <button
              key={fault}
              className={
                selectedFaults.includes(fault)
                ? "selected-fault"
                : "fault-btn"
              }
              onClick={() => toggleFault(fault)}
            >

              {fault}

            </button>

          ))

        }

      </div>

      <p className="selected-text">

        Selected Faults:
        {" "}
        {selectedFaults.join(", ") || "None"}

      </p>

    </div>

    {/* STEP 2 */}

    <div className="section">

      <h2>2. Run MATLAB Simulation</h2>

      <button
        className="main-btn"
        onClick={runSimulation}
      >

        Run Simulation

      </button>

      <p>{simulationStatus}</p>

    </div>

    {/* STEP 3 */}

    <div className="section">

      <h2>3. Upload Existing CSV File</h2>

      <input
        type="file"
        onChange={(e) =>
          setUploadedFile(e.target.files[0])
        }
      />

      {

        uploadedFile &&

        <p>{uploadedFile.name}</p>

      }

      <button
        className="main-btn"
        onClick={uploadFile}
      >

        Upload Dataset

      </button>

    </div>

    {/* STEP 4 */}

    <div className="section">

      <h2>4. Select Current Operating Day</h2>

      <input
        type="range"
        min="1"
        max="20"
        value={currentDay}
        onChange={(e) =>
          setCurrentDay(e.target.value)
        }
        className="slider"
      />

      <p>

        Current Day:
        {" "}
        {currentDay}

      </p>

    </div>

    {/* STEP 5 */}

    <div className="section">

      <h2>5. Run Prediction</h2>

      <button
        className="main-btn"
        onClick={makePrediction}
      >

        Make Prediction

      </button>

    </div>

    {/* GRAPH */}

    {

      graphUrl && (

        <div className="section">

          <h2>RUL Prediction Graph</h2>

          <img
            src={graphUrl}
            alt="Simulation Graph"
            className="graph-image"
          />

        </div>

      )

    }

    {/* RESULTS */}

    {

      predictionResult && (

        <div className="prediction-results">

          <div className="result-card">

            <h3>Detected Fault</h3>

            <p>{predictionResult.fault}</p>

          </div>

          <div className="result-card">

            <h3>Remaining Useful Life</h3>

            <p>{predictionResult.rul} Days</p>

          </div>

          <div className="result-card">

            <h3>Current Day</h3>

            <p>{predictionResult.current_day}</p>

          </div>

        </div>

      )

    }

    {/* DATASET */}

    {

      dataset.length > 0 && (

        <div className="section">

          <h2>Generated Dataset</h2>

          <table>

            <thead>

              <tr>

                <th>Time</th>
                <th>Pressure</th>
                <th>Temperature</th>
                <th>Fault</th>

              </tr>

            </thead>

            <tbody>

              {

                dataset.map((item, index) => (

                  <tr key={index}>

                    <td>{item.time}</td>
                    <td>{item.pressure}</td>
                    <td>{item.temperature}</td>
                    <td>{item.fault}</td>

                  </tr>

                ))

              }

            </tbody>

          </table>

        </div>

      )

    }

  </div>

);

}

export default App;