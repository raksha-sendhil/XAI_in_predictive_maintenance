%% Multi-Class Fault Detection - Lifecycle-Based Dataset Generation
% Generates ~2000 simulations (~500 per fault class) using a triplex
% reciprocating pump Simulink model. Each simulation represents one "day"
% in a machine's lifecycle, with fault severity progressing exponentially
% from healthy baseline to maximum fault level.
%
% Fault classes:
% 0 - No fault
% 1 - Leak fault (leak_cyl_area_WKSP increases over lifecycle)
% 2 - Blockage fault (block_in_factor_WKSP decreases over lifecycle)
% 3 - Bearing fault (bearing_fault_frict_WKSP increases over lifecycle)
%
% Follows the documentation ensemble workflow:
% 1. Build Simulink.SimulationInput array
% 2. generateSimulationEnsemble -> stores .mat files in ./Data
% 3. simulationEnsembleDatastore -> used later for feature extraction
%
% Metadata (machine_id, current_day, total_lifecycle_days, fault_class,
% severity values) is embedded as setVariable entries in each
% SimulationInput so it is recoverable from the ensemble identically to
% how fault variable values are recovered in the documentation.

%% -----------------------------------------------------------------------
% SECTION 1 – Model Setup
% -----------------------------------------------------------------------
% Unzip supporting files and load pump parameters (same as documentation)
unzip('pdmRecipPump_supportingfiles.zip')

% Load pump parameters into workspace
pdmRecipPump_Parameters % loads TRP_Par struct
CAT_Pump_1051_DataFile_imported

% Open the model
mdl = 'pdmRecipPump';
open_system(mdl)

%% -----------------------------------------------------------------------
% SECTION 2 – Fault Parameter Ranges (UNCHANGED from documentation)
% -----------------------------------------------------------------------
% These define the physical min/max for each fault type.
% "Healthy baseline" = index-1 value of each linspace (no-fault level).
% "Maximum fault" = last value of each linspace.
numParValues = 10;

% Leak: area increases from ~0 to 0.036 * Max_Area
leak_area_set_factor = linspace(0.00, 0.036, numParValues);
leak_area_set = leak_area_set_factor * TRP_Par.Check_Valve.In.Max_Area;
leak_area_set = max(leak_area_set, 1e-9); % area cannot be 0

% Blockage: blocking factor decreases from 0.8 (healthy) to 0.53 (severe)
blockinfactor_set = linspace(0.8, 0.53, numParValues);

% Bearing: friction increases from 0 (healthy) to 6e-4 (severe)
bearingfactor_set = linspace(0, 6e-4, numParValues);

% Convenient named constants for healthy baselines and fault extremes
LEAK_MIN = leak_area_set(1); % 1e-9 (healthy)
LEAK_MAX = leak_area_set(end); % 0.036 * Max_Area
BLOCK_MIN = blockinfactor_set(end); % 0.53 (most severe / minimum value)
BLOCK_MAX = blockinfactor_set(1); % 0.80 (healthy / maximum value)
BEARING_MIN = bearingfactor_set(1); % 0 (healthy)
BEARING_MAX = bearingfactor_set(end); % 6e-4

%% -----------------------------------------------------------------------
% SECTION 3 – Dataset Size & Machine Configuration
% -----------------------------------------------------------------------
% Target: ~500 rows per fault class, ~2000 rows total.
% Each machine contributes (lifecycle_days) rows.
% With lifecycles in [30,45] days, mean ~37 days.
% 500 / 37 ≈ 14 machines per class → 14 * 4 = 56 machines total.
TARGET_ROWS_PER_CLASS = 500;
NUM_CLASSES = 4; % no_fault, leak, blockage, bearing
MACHINES_PER_CLASS = 1; % adjust if needed to hit target

% Lifecycle range (days)
LC_MIN = 5;
LC_MAX = 7;

% Fault class labels (stored as integers in SimulationInput)
CLASS_NO_FAULT = 0;
CLASS_LEAK = 1;
CLASS_BLOCKAGE = 2;
CLASS_BEARING = 3;

% Global random seed for reproducibility
rng('default');

%% -----------------------------------------------------------------------
% SECTION 4 – Per-Machine Parameter Generation
% -----------------------------------------------------------------------
% Pre-allocate machine parameter struct array
totalMachines = NUM_CLASSES * MACHINES_PER_CLASS; % 56 for final dataset
machines = struct();
machineIdx = 0;

for classID = [CLASS_NO_FAULT, CLASS_LEAK, CLASS_BLOCKAGE, CLASS_BEARING]
    for m = 1:MACHINES_PER_CLASS
        machineIdx = machineIdx + 1;
        % Unique machine identifier
        machines(machineIdx).machine_id = machineIdx;
        machines(machineIdx).fault_class = classID;
        % Randomly assign lifecycle length (integer days, good diversity)
        machines(machineIdx).total_lifecycle_days = randi([LC_MIN, LC_MAX]);
        % Randomly assign degradation curve shape exponent per machine
        % alpha in [1.5, 3]: higher = slower early degradation, faster late
        machines(machineIdx).alpha = 1.5 + 1.5 * rand();
    end
end

%% -----------------------------------------------------------------------
% SECTION 5 – Severity Progression Functions
% -----------------------------------------------------------------------
% Formulation:
% normalized_time = current_day / total_lifecycle_days
%
% Leak & Bearing (increasing):
% severity = min + (max - min) * (normalized_time ^ alpha)
%
% Blockage (decreasing):
% severity = max - (max - min) * (normalized_time ^ alpha)
%
% Physical parameters follow a perfectly smooth curve. 
% Variance is introduced strictly via Simulink's noise_seed_offset_WKSP.

%% -----------------------------------------------------------------------
% SECTION 6 – Build Simulink.SimulationInput Array
% -----------------------------------------------------------------------
fprintf('=== Building SimulationInput array ===\n');

% Count total simulations
totalSims = 0;
for k = 1:totalMachines
    totalSims = totalSims + machines(k).total_lifecycle_days;
end
fprintf('Total simulations planned: %d\n', totalSims);

% Pre-allocate SimulationInput array
simInput(totalSims) = Simulink.SimulationInput(mdl);
simCounter = 0; % global index into simInput array

for k = 1:totalMachines
    mID = machines(k).machine_id;
    fClass = machines(k).fault_class;
    totalDays = machines(k).total_lifecycle_days;
    alpha = machines(k).alpha;

    fprintf(' Machine %d | Class %d | Lifecycle %d days | alpha=%.3f\n', ...
        mID, fClass, totalDays, alpha);

    for day = 1:totalDays
        simCounter = simCounter + 1;

        % ----------------------------------------------------------------
        % Compute pure severity for this day (monotonic by definition)
        % ----------------------------------------------------------------
        norm_t = day / totalDays; % 0 < norm_t <= 1; at day==totalDays => 1
        % Base exponential curve
        base_curve = norm_t ^ alpha;
        
        % Small smooth variation (controlled)
        variation = 0.02 * randn();   % ~2% variation
        
        norm_t_eff = base_curve * (1 + variation);
        
        % Keep within valid range
        norm_t_eff = max(0, min(1, norm_t_eff));

        if fClass == CLASS_NO_FAULT
            leak_sev = LEAK_MIN;
            block_sev = BLOCK_MAX;
            bearing_sev = BEARING_MIN;

        elseif fClass == CLASS_LEAK
            leak_sev = LEAK_MIN + (LEAK_MAX - LEAK_MIN) * norm_t_eff;
            block_sev = BLOCK_MAX;
            bearing_sev = BEARING_MIN;

        elseif fClass == CLASS_BLOCKAGE
            block_sev = BLOCK_MAX - (BLOCK_MAX - BLOCK_MIN) * norm_t_eff;
            leak_sev = LEAK_MIN;
            bearing_sev = BEARING_MIN;

        elseif fClass == CLASS_BEARING
            bearing_sev = BEARING_MIN + (BEARING_MAX - BEARING_MIN) * norm_t_eff;
            leak_sev = LEAK_MIN;
            block_sev = BLOCK_MAX;
        end

        % --- Force exact failure at last day ---
        if day == totalDays
            if fClass == CLASS_LEAK
                leak_sev = LEAK_MAX;
            elseif fClass == CLASS_BLOCKAGE
                block_sev = BLOCK_MIN;
            elseif fClass == CLASS_BEARING
                bearing_sev = BEARING_MAX;
            end
        end
        % ----------------------------------------------------------------
        % Build SimulationInput for this day
        % ----------------------------------------------------------------
        si = Simulink.SimulationInput(mdl);

        % --- Physical fault parameters (same variables as documentation) ---
        si = setVariable(si, 'leak_cyl_area_WKSP', leak_sev);
        si = setVariable(si, 'block_in_factor_WKSP', block_sev);
        si = setVariable(si, 'bearing_fault_frict_WKSP', bearing_sev);

        % --- Unique noise seed per simulation (as per documentation) ---
        si = setVariable(si, 'noise_seed_offset_WKSP', simCounter - 1);

        % --- Metadata variables (recoverable from ensemble later) ---
        si = setVariable(si, 'machine_id_WKSP', mID);
        si = setVariable(si, 'total_lifecycle_days_WKSP', totalDays);
        si = setVariable(si, 'current_day_WKSP', day);
        si = setVariable(si, 'fault_class_WKSP', fClass);

        % Severity values stored for traceability
        si = setVariable(si, 'leak_severity_WKSP', leak_sev);
        si = setVariable(si, 'blockage_severity_WKSP', block_sev);
        si = setVariable(si, 'bearing_severity_WKSP', bearing_sev);

        % --- Explicit labels for datastore (CRITICAL) ---
        si = setVariable(si, 'LeakFault', leak_sev);
        si = setVariable(si, 'BlockageFault', block_sev);
        si = setVariable(si, 'BearingFault', bearing_sev);

        simInput(simCounter) = si;
    end % day loop
end % machine loop

fprintf('SimulationInput array built: %d entries.\n', simCounter);

%% -----------------------------------------------------------------------
% SECTION 7 – Run Simulations & Store Results
% -----------------------------------------------------------------------
% Prepare output folder
if isfolder('./Data')
    delete('./Data/*.mat')
else
    mkdir('./Data')
end

fprintf('=== Running %d simulations ===\n', simCounter);

% Run all simulations at once. generateSimulationEnsemble handles memory 
% efficiently and writes individual MAT files to the disk sequentially.
[ok, e] = generateSimulationEnsemble( ...
    simInput, ...
    fullfile('.', 'Data'), ...
    'UseParallel', false, ...
    'ShowProgress', true);

if all(ok)
    fprintf('All simulations completed successfully.\n');
else
    fprintf('WARNING: %d simulations failed. Check the "e" variable for details.\n', sum(~ok));
end

%% -----------------------------------------------------------------------
% SECTION 8 – Create simulationEnsembleDatastore
% -----------------------------------------------------------------------
% Create the ensemble datastore from the stored results.
% This follows the documentation exactly.
ens = simulationEnsembleDatastore(fullfile('.', 'Data'));

fprintf('=== Ensemble datastore created ===\n');
fprintf('Available data variables:\n');
disp(ens.DataVariables)

fprintf('\nDataset generation complete.\n');
fprintf('Ensemble stored in: %s\n', fullfile(pwd, 'Data'));