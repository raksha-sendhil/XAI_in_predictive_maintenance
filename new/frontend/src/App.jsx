
import { useState } from "react";
import "./App.css";

function App() {

  const [selectedFaults, setSelectedFaults] = useState([]);
  const [dataset, setDataset] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [currentDay, setCurrentDay] = useState(1);
  const [simulationStatus, setSimulationStatus] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [graphUrl, setGraphUrl] = useState("");
  const [rul, setRul] = useState(0);
  const [showSimulationImage, setShowSimulationImage] = useState(false);

  const faults = [
    "BearingFault",
    "BlockingFault",
    "LeakFault",
    "CavitationFault"
  ];

  const toggleFault = (fault) => {

    if (selectedFaults.includes(fault)) {

      setSelectedFaults(
        selectedFaults.filter((f) => f !== fault)
      );

    }

    else {

      if (selectedFaults.length < 3) {

        setSelectedFaults([
          ...selectedFaults,
          fault
        ]);

      }

    }

  };

  const runSimulation = async () => {

    console.log("Simulation button clicked");

    if (selectedFaults.length === 0) {

      alert("Please select at least one fault");

      return;

    }

    setLoading(true);

    setShowSimulationImage(true);

    setProgress(0);

    setSimulationStatus(
      "Initializing MATLAB Simulation..."
    );

    try {

      for (let i = 0; i <= 100; i += 10) {

        setProgress(i);

        await new Promise(
          (resolve) => setTimeout(resolve, 300)
        );

      }

      const response = await fetch(
        "http://127.0.0.1:5000/simulate",
        {

          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({

            faults: selectedFaults,

            currentDay: Number(currentDay)

          })

        }
      );

      const data = await response.json();

      console.log(data);

      setDataset(data.dataset);

      setRul(data.minimum_rul);

      setGraphUrl(
        `http://127.0.0.1:5000/graph?${new Date().getTime()}`
      );

      setSimulationStatus(
        "Simulation Completed Successfully"
      );

    }

    catch (error) {

      console.log(error);

      setSimulationStatus(
        "Simulation Failed"
      );

    }

    setLoading(false);

  };

  const uploadFile = async () => {

    if (!uploadedFile) {

      alert("Please select a file");

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

      setRul(data.minimum_rul);

      setGraphUrl(
        `http://127.0.0.1:5000/graph?${new Date().getTime()}`
      );

    }

    catch(error) {

      console.log(error);

    }

  };

  const makePrediction = async () => {

    console.log("Prediction button clicked");

    const response = await fetch(
      "http://127.0.0.1:5000/predict",
      {
        method: "POST"
      }
    );

    const data = await response.json();

    console.log(data);

    setPredictionResult(data);

  };

  return (

    <div className="App">

     
      <h1 className="main-title">
        ⚙️ Triplex Pump Predictive Maintenance
      </h1>

      <p className="subtitle">
        Fault Detection · Severity Analysis · RUL Prediction
      </p>



      <hr />

      <div className="section">

        <h2>Select Fault Types</h2>

        {

          faults.map((fault) => (

           
          <button
            key={fault}
            className={
              selectedFaults.includes(fault)
              ? "selected-fault"
              : ""
            }
            onClick={() => toggleFault(fault)}
          >



              {fault}

            </button>

          ))

        }

        <p>

          Selected Faults:
          {selectedFaults.join(", ")}

        </p>

      </div>

      <hr />

      <h2>Select Current Operating Day</h2>

      <input
        type="range"
        min="1"
        max="20"
        value={currentDay}
        onChange={(e) =>
          setCurrentDay(e.target.value)
        }
      />

      <p>

        Current Day: {currentDay}

      </p>

      <hr />

      <h2>Simulation Controls</h2>

      <button onClick={runSimulation}>

        Run Simulations

      </button>

      <p>{simulationStatus}</p>

      {

        loading &&

        <div>

          <progress
            value={progress}
            max="100"
            style={{
              width:"700px",
              height:"30px"
            }}
          />

          <p>

            Simulation Progress: {progress}%

          </p>

        </div>

      }

      <hr />

      <h2>Simulation Graphs</h2>

      {

        showSimulationImage &&

        <img
          src="/pump.png"
          width="500"
          alt="Pump"
        />

      }

      {
      <img
        src={graphUrl}
        alt="Simulation Graph"
        className="graph-image"
      />}



      <h2>

        Estimated RUL: {rul} Days

      </h2>

      <hr />

      <h2>Generated Dataset</h2>

      <table border="1">

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

            dataset &&
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

      <hr />

      <h2>Upload Existing Simulation File</h2>

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

      <button onClick={uploadFile}>

        Upload Dataset

      </button>

      <hr />

      <h2>Prediction Section</h2>

      <button onClick={makePrediction}>

        Make Prediction

      </button>

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

    </div>

  );

}

export default App;

