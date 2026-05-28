%% DEMO LAST-DAY RUNNER + ONE-ROW CSV EXPORT
% This script:
%
% 1. Loads ONE last-day-only simInput MAT file
% 2. Runs exactly ONE simulation
% 3. Extracts the SAME features as your ML pipeline
% 4. Exports a ONE-ROW CSV
%
% Works for ANY fault class MAT file:
%   - Classes 0-3
%   - Classes 4-7
%
% Example input files:
%   Machine_1_Class_0_LASTDAY.mat
%   Machine_2_Class_5_LASTDAY.mat
%   etc.

%% -----------------------------------------------------------------------
% SECTION 1 – USER SETTINGS
% -----------------------------------------------------------------------
clc;
clear;

dataFile = 'Machine_4_Class_3_LASTDAY.mat';
%this directory will have 8 such .mat files for the 8 different fault classes, all with "LASTDAY" in the name

%% -----------------------------------------------------------------------
% SECTION 2 – MODEL SETUP
% -----------------------------------------------------------------------
if exist('pdmRecipPump_supportingfiles.zip', 'file') == 2

    unzip('pdmRecipPump_supportingfiles.zip');

end

pdmRecipPump_Parameters
CAT_Pump_1051_DataFile_imported

mdl = 'pdmRecipPump';

load_system(mdl)
open_system(mdl)

%% -----------------------------------------------------------------------
% SECTION 3 – LOAD SIMINPUT
% -----------------------------------------------------------------------
fprintf('Loading: %s\n', dataFile);

s = load(dataFile);

assert(isfield(s, 'simInputBatch'), ...
    'MAT file missing simInputBatch');

simInputBatch = s.simInputBatch;

assert(numel(simInputBatch) == 1, ...
    'This script expects ONLY ONE simulation.');

%% -----------------------------------------------------------------------
% SECTION 4 – CREATE OUTPUT FOLDER
% -----------------------------------------------------------------------
[~, baseName, ~] = fileparts(dataFile);

outputRoot = fullfile('.', ['DEMO_OUTPUT_' baseName]);

if ~isfolder(outputRoot)

    mkdir(outputRoot);

end

%% -----------------------------------------------------------------------
% SECTION 5 – RUN SINGLE SIMULATION
% -----------------------------------------------------------------------
fprintf('\n=== Running ONE simulation ===\n');

[ok, e] = generateSimulationEnsemble( ...
    simInputBatch, ...
    outputRoot, ...
    'UseParallel', false, ...
    'ShowProgress', true);

%% CHECK FAILURE
if ~ok

    fprintf('\nSIMULATION FAILED\n');

    try

        disp(e(1).Error.message);

    catch

        disp('Unknown error');

    end

    return;

end

fprintf('Simulation completed successfully.\n');

%% -----------------------------------------------------------------------
% SECTION 6 – CREATE DATASTORE
% -----------------------------------------------------------------------
ens = simulationEnsembleDatastore(outputRoot);

ens.DataVariables = ...
    ["logsout_sm_pump_triplex"; "SimulationInput"];

ens.SelectedVariables = ...
    ["logsout_sm_pump_triplex"; "SimulationInput"];

reset(ens);

%% -----------------------------------------------------------------------
% SECTION 7 – FEATURE + LABEL DEFINITIONS
% -----------------------------------------------------------------------
featureNames = ...
    ["fPeak"; ...
     "pLow"; ...
     "pMid"; ...
     "pHigh"; ...
     "pKurtosis"; ...
     "qMean"; ...
     "qVar"; ...
     "qSkewness"; ...
     "qKurtosis"; ...
     "qPeak2Peak"; ...
     "qCrest"; ...
     "qRMS"; ...
     "qMAD"; ...
     "qCSRange"];

labelNames = ...
    ["LeakFault"; ...
     "BlockingFault"; ...
     "BearingFault"; ...
     "FaultClass"; ...
     "machine_id"; ...
     "total_lifecycle_days"; ...
     "current_day"];

ens.DataVariables = ...
    [ens.DataVariables; featureNames];

ens.ConditionVariables = labelNames;

%% -----------------------------------------------------------------------
% SECTION 8 – FEATURE EXTRACTION
% -----------------------------------------------------------------------
fprintf('\n=== Extracting features ===\n');

data = read(ens);

[flow, flowP, flowF, faultValues] = preprocess(data);

feat = extractCI(flow, flowP, flowF);

dataToWrite = [faultValues, feat];

writeToLastMemberRead(ens, dataToWrite{:});

fprintf('Feature extraction complete.\n');

%% -----------------------------------------------------------------------
% SECTION 9 – GATHER FINAL TABLE
% -----------------------------------------------------------------------
ens.SelectedVariables = ...
    [featureNames; labelNames];

reset(ens);

finalDataset = gather(tall(ens));

%% -----------------------------------------------------------------------
% SECTION 10 – EXPORT CSV
% -----------------------------------------------------------------------
csvName = sprintf('%s_one_row.csv', baseName);

writetable(finalDataset, csvName);

fprintf('\n=====================================\n');
fprintf('SUCCESS: One-row CSV generated.\n');
fprintf('CSV File: %s\n', csvName);
fprintf('=====================================\n');

disp(finalDataset)

%% -----------------------------------------------------------------------
% SUPPORTING FUNCTIONS
% -----------------------------------------------------------------------
function [flow, flowSpectrum, flowFrequencies, faultValues] = preprocess(data)

    %% EXTRACT FLOW SIGNAL
    logs = data.logsout_sm_pump_triplex{1};

    flow = logs.get('qOut_meas').Values;

    %% REMOVE STARTUP TRANSIENTS
    tMin = seconds(0.8);

    flow = flow(flow.Time >= tMin,:);

    %% RESAMPLE SIGNAL
    flow = retime( ...
        flow, ...
        'regular', ...
        'linear', ...
        'TimeStep', ...
        seconds(1e-3));

    %% REMOVE MEAN
    fA = flow;

    fA.Data = fA.Data - mean(fA.Data);

    %% COMPUTE SPECTRUM
    [flowSpectrum, flowFrequencies] = ...
        pspectrum( ...
            fA, ...
            'FrequencyLimits', ...
            [2 250]);

    %% READ SIMULATIONINPUT VARIABLES
    simin = data.SimulationInput{1};

    vars = {simin.Variables.Name};

    %% EXTRACT FAULT VALUES
    idx = strcmp(vars, 'leak_cyl_area_WKSP');
    LeakFault = simin.Variables(idx).Value;

    idx = strcmp(vars, 'block_in_factor_WKSP');
    BlockingFault = simin.Variables(idx).Value;

    idx = strcmp(vars, 'bearing_fault_frict_WKSP');
    BearingFault = simin.Variables(idx).Value;

    %% EXTRACT METADATA
    idx = strcmp(vars, 'fault_class_WKSP');
    FaultClass = simin.Variables(idx).Value;

    idx = strcmp(vars, 'machine_id_WKSP');
    machine_id = simin.Variables(idx).Value;

    idx = strcmp(vars, 'total_lifecycle_days_WKSP');
    total_lifecycle_days = simin.Variables(idx).Value;

    idx = strcmp(vars, 'current_day_WKSP');
    current_day = simin.Variables(idx).Value;

    %% PACKAGE LABELS
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

    %% PEAK FREQUENCY
    pMax = max(flowP);

    fPeak = flowF(flowP == pMax);

    %% LOW BAND POWER
    fRange = flowF >= 10 & flowF <= 20;

    pLow = sum(flowP(fRange));

    %% MID BAND POWER
    fRange = flowF >= 40 & flowF <= 60;

    pMid = sum(flowP(fRange));

    %% HIGH BAND POWER
    fRange = flowF >= 100;

    pHigh = sum(flowP(fRange));

    %% SPECTRAL KURTOSIS
    [pKur, fKur] = pkurtosis(flow);

    pKur = fKur(pKur == max(pKur));

    %% CUMULATIVE SUM RANGE
    csFlow = cumsum(flow.Data);

    csFlowRange = max(csFlow) - min(csFlow);

    %% PACKAGE FEATURES
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