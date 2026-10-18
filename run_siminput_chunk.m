%% Multi-Class Fault Detection - WORKER SCRIPT (HARDENED)
% Runs an assigned SimulationInput chunk in small batches to avoid crashes
% from memory/disk pressure. Safe to restart: completed batches are skipped.

%% 0. USER SETTINGS
% ---> CHANGE THIS FILENAME to the one you were assigned! <---
dataFile = 'Part1_You_simInput.mat';

% Run in smaller batches to reduce memory spikes and long single-run failures
batchSize = 25;
pauseBetweenBatchesSec = 1;
resumeIfStateExists = true;

%% 1. Model Setup
if exist('pdmRecipPump_supportingfiles.zip', 'file') == 2
    unzip('pdmRecipPump_supportingfiles.zip');
end
pdmRecipPump_Parameters
CAT_Pump_1051_DataFile_imported
mdl = 'pdmRecipPump';
load_system(mdl)

%% 2. Load Assigned Configuration
fprintf('Loading configuration: %s\n', dataFile);
s = load(dataFile);
assert(isfield(s, 'simInputBatch'), 'MAT file does not contain variable "simInputBatch".');
simInputBatch = s.simInputBatch;
assert(isa(simInputBatch, 'Simulink.SimulationInput'), ...
    '"simInputBatch" must be Simulink.SimulationInput array.');

if isfield(s, 'chunkMeta')
    fprintf('Chunk metadata detected: %d rows planned.\n', s.chunkMeta.n_rows);
end

numSims = numel(simInputBatch);
fprintf('Total simulations in this chunk: %d\n', numSims);

%% 3. Setup Output + State Paths
[~, folderName, ~] = fileparts(dataFile);
outputRoot = fullfile('.', ['Data_', folderName]);
batchRoot = fullfile(outputRoot, 'batches');
flatOutputDir = fullfile(outputRoot, 'flat');
stateFile = fullfile(outputRoot, 'run_state.mat');
failureLogFile = fullfile(outputRoot, 'failure_log.txt');

if ~isfolder(outputRoot), mkdir(outputRoot); end
if ~isfolder(batchRoot), mkdir(batchRoot); end
if ~isfolder(flatOutputDir), mkdir(flatOutputDir); end

%% 4. Build/Load Run State
numBatches = ceil(numSims / batchSize);
completedBatches = false(1, numBatches);
failedGlobalIndices = [];
failedMessages = {};

if resumeIfStateExists && isfile(stateFile)
    stateData = load(stateFile);
    if isfield(stateData, 'completedBatches') && numel(stateData.completedBatches) == numBatches
        completedBatches = stateData.completedBatches;
    end
    if isfield(stateData, 'failedGlobalIndices')
        failedGlobalIndices = stateData.failedGlobalIndices;
    end
    if isfield(stateData, 'failedMessages')
        failedMessages = stateData.failedMessages;
    end
    fprintf('Resume mode: %d/%d batches already complete.\n', nnz(completedBatches), numBatches);
end

%% 5. Run ONE Batch per Execution (Crash-Proof Mode)
fprintf('=== Running %d simulations in %d batches (batch size = %d) ===\n', ...
    numSims, numBatches, batchSize);

ranBatch = false;

for b = 1:numBatches
    if completedBatches(b)
        fprintf('Skipping batch %d/%d (already complete).\n', b, numBatches);
        continue;
    end

    % ---- RUN ONLY THE FIRST INCOMPLETE BATCH ----
    idxStart = (b - 1) * batchSize + 1;
    idxEnd = min(b * batchSize, numSims);
    localGlobalIndices = idxStart:idxEnd;
    simInputLocal = simInputBatch(localGlobalIndices);

    batchDir = fullfile(batchRoot, sprintf('batch_%03d', b));
    if ~isfolder(batchDir), mkdir(batchDir); end

    fprintf('\nRunning batch %d/%d | Sim %d to %d\n', b, numBatches, idxStart, idxEnd);

    [ok, e] = generateSimulationEnsemble( ...
        simInputLocal, ...
        batchDir, ...
        'UseParallel', false, ...
        'ShowProgress', true);

    % ---- SAVE BATCH REPORT ----
    batchReportFile = fullfile(batchDir, 'batch_report.mat');
    save(batchReportFile, 'ok', 'e', 'localGlobalIndices');

    % ---- HANDLE FAILURES ----
    failedLocal = find(~ok);
    if isempty(failedLocal)
        completedBatches(b) = true;
        fprintf('Batch %d complete with no failures.\n', b);
    else
        fid = fopen(failureLogFile, 'a');
        if fid ~= -1
            fprintf(fid, '\n=== Batch %d failures (%s) ===\n', b, datestr(now));
        end

        for k = 1:numel(failedLocal)
            localIdx = failedLocal(k);
            globalIdx = localGlobalIndices(localIdx);
            try
                msg = e(localIdx).Error.message;
            catch
                msg = '(unknown error)';
            end
            failedGlobalIndices(end+1) = globalIdx; %#ok<AGROW>
            failedMessages{end+1} = msg; %#ok<AGROW>
            fprintf('FAILED sim #%d: %s\n', globalIdx, msg);
            if fid ~= -1
                fprintf(fid, 'FAILED sim #%d: %s\n', globalIdx, msg);
            end
        end

        if fid ~= -1
            fclose(fid);
        end

        warning('Batch %d had failures. Fix before continuing.', b);

        save(stateFile, 'completedBatches', 'failedGlobalIndices', 'failedMessages', ...
            'dataFile', 'batchSize', 'numSims', 'numBatches');

        return;
    end

    % ---- SAVE STATE AFTER SUCCESS ----
    save(stateFile, 'completedBatches', 'failedGlobalIndices', 'failedMessages', ...
        'dataFile', 'batchSize', 'numSims', 'numBatches');

    fprintf('Batch %d saved successfully.\n', b);
    pause(pauseBetweenBatchesSec);
    fprintf('IMPORTANT: Restart MATLAB before running again.\n');

    ranBatch = true;
    break; % <-- KEY: stop after ONE batch
end

% ---- IF ALL BATCHES DONE → FLATTEN ----
if ~ranBatch
    fprintf('\nAll batches already completed. Proceeding to flatten...\n');

    flatIndex = 0;
    manifest = cell(0, 2);

    for b = 1:numBatches
        batchDir = fullfile(batchRoot, sprintf('batch_%03d', b));
        mats = dir(fullfile(batchDir, '*.mat'));

        for i = 1:numel(mats)
            if strcmpi(mats(i).name, 'batch_report.mat')
                continue;
            end

            src = fullfile(mats(i).folder, mats(i).name);

            % ---- VALIDATE FILE ----
            try
                tmp = load(src);  %#ok<NASGU> loaded into struct to avoid workspace pollution
            catch
                fprintf('Skipping corrupt file: %s\n', mats(i).name);
                continue;
            end

            % ---- COPY ONLY VALID FILE ----
            flatIndex = flatIndex + 1;
            dstName = sprintf('sim_%05d.mat', flatIndex);
            dst = fullfile(flatOutputDir, dstName);

            copyfile(src, dst);
            manifest(end+1, :) = {src, dst}; %#ok<AGROW>
        end
    end

    save(fullfile(outputRoot, 'flatten_manifest.mat'), 'manifest');

    fprintf('\nSUCCESS: Completed %d/%d simulations.\n', numSims, numSims);
    fprintf('Flat files ready in: %s\n', flatOutputDir);
end
