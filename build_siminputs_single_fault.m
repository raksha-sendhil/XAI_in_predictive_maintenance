%% Multi-Class Fault Detection - MASTER SCRIPT (Configuration Generation)
% Run this ONCE to generate the SimulationInput arrays and export them into 5 chunks.

% -----------------------------------------------------------------------
% SECTION 1 – Model Setup
% -----------------------------------------------------------------------
unzip('pdmRecipPump_supportingfiles.zip')
pdmRecipPump_Parameters 
CAT_Pump_1051_DataFile_imported
mdl = 'pdmRecipPump';
load_system(mdl) % Use load_system for background processing

% -----------------------------------------------------------------------
% SECTION 2 – Fault Parameter Ranges
% -----------------------------------------------------------------------
numParValues = 10;
leak_area_set_factor = linspace(0.00, 0.036, numParValues);
leak_area_set = leak_area_set_factor * TRP_Par.Check_Valve.In.Max_Area;
leak_area_set = max(leak_area_set, 1e-9); 
blockinfactor_set = linspace(0.8, 0.53, numParValues);
bearingfactor_set = linspace(0, 6e-4, numParValues);

LEAK_MIN = leak_area_set(1); 
LEAK_MAX = leak_area_set(end); 
BLOCK_MIN = blockinfactor_set(end); 
BLOCK_MAX = blockinfactor_set(1); 
BEARING_MIN = bearingfactor_set(1); 
BEARING_MAX = bearingfactor_set(end); 

% -----------------------------------------------------------------------
% SECTION 3 – Dataset Size & Machine Configuration
% -----------------------------------------------------------------------
TARGET_ROWS_PER_CLASS = 500;
NUM_CLASSES = 4; 
% 14 machines * ~37.5 avg days * 4 classes = ~2100 rows total
MACHINES_PER_CLASS = 14; 
LC_MIN = 30;
LC_MAX = 45;

CLASS_NO_FAULT = 0; CLASS_LEAK = 1; CLASS_BLOCKAGE = 2; CLASS_BEARING = 3;
rng('default');

% -----------------------------------------------------------------------
% SECTION 4 – Per-Machine Parameter Generation
% -----------------------------------------------------------------------
totalMachines = NUM_CLASSES * MACHINES_PER_CLASS; % 56 total machines
machines = struct();
machineIdx = 0;
for classID = [CLASS_NO_FAULT, CLASS_LEAK, CLASS_BLOCKAGE, CLASS_BEARING]
    for m = 1:MACHINES_PER_CLASS
        machineIdx = machineIdx + 1;
        machines(machineIdx).machine_id = machineIdx;
        machines(machineIdx).fault_class = classID;
        machines(machineIdx).total_lifecycle_days = randi([LC_MIN, LC_MAX]);
        machines(machineIdx).alpha = 1.5 + 1.5 * rand();
    end
end

% -----------------------------------------------------------------------
% SECTION 5 & 6 – Build SimulationInput Array
% -----------------------------------------------------------------------
fprintf('=== Building SimulationInput array ===\n');
totalSims = sum([machines.total_lifecycle_days]);
fprintf('Total simulations planned: %d\n', totalSims);

simInput(totalSims) = Simulink.SimulationInput(mdl);
simCounter = 0; 
machine_tracker = zeros(1, totalSims); % NEW: Tracks machine ID for clean splitting

for k = 1:totalMachines
    mID = machines(k).machine_id;
    fClass = machines(k).fault_class;
    totalDays = machines(k).total_lifecycle_days;
    alpha = machines(k).alpha;
    
    prev_norm_t_eff = 0; 
    
    for day = 1:totalDays
        simCounter = simCounter + 1;
        machine_tracker(simCounter) = mID; % Log the machine ID
        
        norm_t_prev = (day - 1) / totalDays;
        norm_t_curr = day / totalDays;
        base_increment = (norm_t_curr ^ alpha) - (norm_t_prev ^ alpha);
        
        variation_multiplier = 0.5 + rand(); 
        noisy_increment = base_increment * variation_multiplier;
        
        norm_t_eff = prev_norm_t_eff + noisy_increment;
        norm_t_eff = min(1, norm_t_eff);
        prev_norm_t_eff = norm_t_eff;
        
        if fClass == CLASS_NO_FAULT
            leak_sev = LEAK_MIN; block_sev = BLOCK_MAX; bearing_sev = BEARING_MIN;
        elseif fClass == CLASS_LEAK
            leak_sev = LEAK_MIN + (LEAK_MAX - LEAK_MIN) * norm_t_eff;
            block_sev = BLOCK_MAX; bearing_sev = BEARING_MIN;
        elseif fClass == CLASS_BLOCKAGE
            block_sev = BLOCK_MAX - (BLOCK_MAX - BLOCK_MIN) * norm_t_eff;
            leak_sev = LEAK_MIN; bearing_sev = BEARING_MIN;
        elseif fClass == CLASS_BEARING
            bearing_sev = BEARING_MIN + (BEARING_MAX - BEARING_MIN) * norm_t_eff;
            leak_sev = LEAK_MIN; block_sev = BLOCK_MAX;
        end
        
        if day == totalDays
            if fClass == CLASS_LEAK, leak_sev = LEAK_MAX;
            elseif fClass == CLASS_BLOCKAGE, block_sev = BLOCK_MIN;
            elseif fClass == CLASS_BEARING, bearing_sev = BEARING_MAX;
            end
        end
        
        si = Simulink.SimulationInput(mdl);
        si = setVariable(si, 'leak_cyl_area_WKSP', leak_sev);
        si = setVariable(si, 'block_in_factor_WKSP', block_sev);
        si = setVariable(si, 'bearing_fault_frict_WKSP', bearing_sev);
        si = setVariable(si, 'noise_seed_offset_WKSP', simCounter - 1);
        si = setVariable(si, 'machine_id_WKSP', mID);
        si = setVariable(si, 'total_lifecycle_days_WKSP', totalDays);
        si = setVariable(si, 'current_day_WKSP', day);
        si = setVariable(si, 'fault_class_WKSP', fClass);
        si = setVariable(si, 'leak_severity_WKSP', leak_sev);
        si = setVariable(si, 'blockage_severity_WKSP', block_sev);
        si = setVariable(si, 'bearing_severity_WKSP', bearing_sev);
        
        si = setVariable(si, 'LeakFault', leak_sev);
        si = setVariable(si, 'BlockingFault', block_sev);
        si = setVariable(si, 'BearingFault', bearing_sev);
        
        simInput(simCounter) = si;
    end 
end 
fprintf('SimulationInput array built: %d entries.\n', simCounter);

% -----------------------------------------------------------------------
% SECTION 7 – Split by Machine and Export
% -----------------------------------------------------------------------
fprintf('=== Splitting into 5 balanced chunks by expected simulation load ===\n');
team = {'Part1_You', 'Part2_Saanvi', 'Part3_Gargi', 'Part4_Adithi', 'Part5_Raksha'};
numTeams = numel(team);

machineDays = [machines.total_lifecycle_days];
[~, sortIdx] = sort(machineDays, 'descend');
teamMachineIds = cell(1, numTeams);
teamLoad = zeros(1, numTeams);

for s = 1:numel(sortIdx)
    m = sortIdx(s);
    [~, t] = min(teamLoad);
    teamMachineIds{t}(end+1) = m; %#ok<AGROW>
    teamLoad(t) = teamLoad(t) + machineDays(m);
end

for i = 1:numTeams
    chunk_indices = ismember(machine_tracker, teamMachineIds{i});
    simInputBatch = simInput(chunk_indices);
    chunkMeta = struct();
    chunkMeta.team_name = team{i};
    chunkMeta.machine_ids = teamMachineIds{i};
    chunkMeta.n_rows = nnz(chunk_indices);
    chunkMeta.recommended_worker_batch_size = 25;
    chunkMeta.global_sim_indices = find(chunk_indices);

    filename = sprintf('%s_simInput.mat', team{i});
    save(filename, 'simInputBatch', 'chunkMeta', '-v7.3');
    fprintf('Saved %s (%d rows, machines: %s)\n', ...
        filename, chunkMeta.n_rows, mat2str(chunkMeta.machine_ids));
end
fprintf('\nDone! Send the .mat files to your team.\n');
