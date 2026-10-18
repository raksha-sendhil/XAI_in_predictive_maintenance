%% Multi-Class Fault Detection - Feature Extraction
% Processes raw simulation data to extract time and frequency domain 
% condition indicators alongside fault labels and metadata for Machine Learning.

% 1. Create datastore and point it to the FLATTENED folder
% ---> UPDATE THIS PATH to where your flattened files are <---
flatDataFolder = './Data_Part1_You_simInput/flat';
ens = simulationEnsembleDatastore(flatDataFolder);

% 1. Explicitly register the variables so the datastore knows they exist
ens.DataVariables = ["logsout_sm_pump_triplex"; "SimulationInput"]; 

% 2. Now you can safely select them
ens.SelectedVariables = ["logsout_sm_pump_triplex"; "SimulationInput"]; 
reset(ens);

% 2. Define the exact features and labels 
featureNames = ["fPeak"; "pLow"; "pMid"; "pHigh"; "pKurtosis"; ...
                "qMean"; "qVar"; "qSkewness"; "qKurtosis"; ...
                "qPeak2Peak"; "qCrest"; "qRMS"; "qMAD"; "qCSRange"];
            
% --- MODIFIED: Added machine_id, total_lifecycle_days, and current_day ---
labelNames = ["LeakFault"; "BlockingFault"; "BearingFault"; "FaultClass"; ...
              "machine_id"; "total_lifecycle_days"; "current_day"];

% Pre-allocate the variables in the datastore
ens.DataVariables = [ens.DataVariables; featureNames]; 
ens.ConditionVariables = labelNames; 

% 3. Loop through the entire datastore to process each simulation
fprintf('=== Starting Feature Extraction ===\n');
simCount = 0;

while hasdata(ens) 
    % Read raw member data
    data = read(ens); 
    
    % Preprocess to remove startup transients and compute spectrum
    [flow, flowP, flowF, faultValues] = preprocess(data); 
    
    % Extract the 14 condition indicators
    feat = extractCI(flow, flowP, flowF); 
    
    % Append features and fault values back to the member file
    dataToWrite = [faultValues, feat]; 
    writeToLastMemberRead(ens, dataToWrite{:}); 
    
    simCount = simCount + 1;
    if mod(simCount, 100) == 0
        fprintf('Processed %d simulations...\n', simCount);
    end
end
fprintf('Feature extraction complete for %d files.\n', simCount);

% 4. Gather the final dataset into memory for ML Training
fprintf('Gathering final dataset into memory...\n');
ens.SelectedVariables = [featureNames; labelNames];
reset(ens);

% Use tall arrays to handle the memory extraction efficiently
finalDataset = gather(tall(ens)); 

% Save the final dataset as a clean MAT file
save('Final_ML_Dataset.mat', 'finalDataset');
fprintf('Dataset saved successfully as Final_ML_Dataset.mat\n');

% === Export to CSV for Python/ML Training ===
writetable(finalDataset, 'dataset.csv');
fprintf('SUCCESS: Dataset successfully exported as dataset.csv!\n');


%% -----------------------------------------------------------------------
% SUPPORTING FUNCTIONS
% -----------------------------------------------------------------------
function [flow, flowSpectrum, flowFrequencies, faultValues] = preprocess(data) 
    % Helper function to preprocess the logged reciprocating pump data. 
    
    % Extract the specific flow sensor from the logsout object
    logs = data.logsout_sm_pump_triplex{1};
    flow = logs.get('qOut_meas').Values;
    
    % Remove the 1st 0.8 seconds of the flow signal to eliminate startup transients 
    tMin = seconds(0.8); 
    flow = flow(flow.Time >= tMin,:); 
    
    % Ensure the flow is sampled at a uniform sample rate 
    flow = retime(flow, 'regular', 'linear', 'TimeStep', seconds(1e-3)); 
    
    % Remove the mean from the flow and compute the flow spectrum 
    fA = flow; 
    fA.Data = fA.Data - mean(fA.Data); 
    [flowSpectrum, flowFrequencies] = pspectrum(fA, 'FrequencyLimits', [2 250]); 
    
    % Find the values of the fault variables from the SimulationInput 
    simin = data.SimulationInput{1}; 
    vars = {simin.Variables.Name}; 
    
    % Extract severity parameters 
    idx = strcmp(vars, 'leak_cyl_area_WKSP'); LeakFault = simin.Variables(idx).Value;
    idx = strcmp(vars, 'block_in_factor_WKSP'); BlockingFault = simin.Variables(idx).Value;
    idx = strcmp(vars, 'bearing_fault_frict_WKSP'); BearingFault = simin.Variables(idx).Value;
    
    % Extract the specific fault class label for ML mapping
    idx = strcmp(vars, 'fault_class_WKSP'); FaultClass = simin.Variables(idx).Value;
    
    % --- MODIFIED: Extract Metadata Parameters ---
    idx = strcmp(vars, 'machine_id_WKSP'); machine_id = simin.Variables(idx).Value;
    idx = strcmp(vars, 'total_lifecycle_days_WKSP'); total_lifecycle_days = simin.Variables(idx).Value;
    idx = strcmp(vars, 'current_day_WKSP'); current_day = simin.Variables(idx).Value;
    
    % Collect the fault values in a cell array (must match labelNames exactly)
    faultValues = {...
        'LeakFault', LeakFault, ...
        'BlockingFault', BlockingFault, ...
        'BearingFault', BearingFault, ...
        'FaultClass', FaultClass, ...
        'machine_id', machine_id, ...
        'total_lifecycle_days', total_lifecycle_days, ...
        'current_day', current_day};
end

function ci = extractCI(flow, flowP, flowF) 
    % Helper function to extract condition indicators from the flow signal and spectrum. 
    
    % Find the frequency of the peak magnitude in the power spectrum. 
    pMax = max(flowP); 
    fPeak = flowF(flowP == pMax); 
    
    % Compute the power in the low frequency range 10-20 Hz. 
    fRange = flowF >= 10 & flowF <= 20; 
    pLow = sum(flowP(fRange)); 
    
    % Compute the power in the mid frequency range 40-60 Hz. 
    fRange = flowF >= 40 & flowF <= 60; 
    pMid = sum(flowP(fRange)); 
    
    % Compute the power in the high frequency range >100 Hz. 
    fRange = flowF >= 100; 
    pHigh = sum(flowP(fRange)); 
    
    % Find the frequency of the spectral kurtosis peak 
    [pKur, fKur] = pkurtosis(flow); 
    pKur = fKur(pKur == max(pKur)); 
    
    % Compute the flow cumulative sum range. 
    csFlow = cumsum(flow.Data); 
    csFlowRange = max(csFlow) - min(csFlow); 
    
    % Collect the feature and feature values in a cell array. 
    ci = {...
        'qMean', mean(flow.Data), ... 
        'qVar', var(flow.Data), ... 
        'qSkewness', skewness(flow.Data), ... 
        'qKurtosis', kurtosis(flow.Data), ... 
        'qPeak2Peak', peak2peak(flow.Data), ... 
        'qCrest', peak2rms(flow.Data), ... 
        'qRMS', rms(flow.Data), ... 
        'qMAD', mad(flow.Data), ... 
        'qCSRange', csFlowRange, ... 
        'fPeak', fPeak(1), ... 
        'pLow', pLow, ... 
        'pMid', pMid, ... 
        'pHigh', pHigh, ... 
        'pKurtosis', pKur(1)}; 
end