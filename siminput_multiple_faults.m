%% Multi-Class Fault Detection - MASTER SCRIPT (Configuration Generation)
% Run this ONCE to generate the SimulationInput arrays and export them into 5 chunks.

%% -----------------------------------------------------------------------
% SECTION 1 – Model Setup
% -----------------------------------------------------------------------
unzip('pdmRecipPump_supportingfiles.zip')
pdmRecipPump_Parameters 
CAT_Pump_1051_DataFile_imported
mdl = 'pdmRecipPump';
load_system(mdl) % Use load_system for background processing

%%
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

CLASS_LEAK_BLOCK = 0; CLASS_BLOCK_BEARING = 1; CLASS_BEARING_LEAK = 2; CLASS_ALL_FAULTS = 3;
rng('default');

% -----------------------------------------------------------------------
% SECTION 4 – Per-Machine Parameter Generation
% -----------------------------------------------------------------------
% -----------------------------------------------------------------------
% SECTION 4 – Per-Machine Parameter Generation
% -----------------------------------------------------------------------
totalMachines = NUM_CLASSES * MACHINES_PER_CLASS; 
machines = struct();
machineIdx = 0;

for classID = [CLASS_LEAK_BLOCK, CLASS_BLOCK_BEARING, CLASS_BEARING_LEAK, CLASS_ALL_FAULTS]
    for m = 1:MACHINES_PER_CLASS
        machineIdx = machineIdx + 1;
        machines(machineIdx).machine_id = machineIdx;
        machines(machineIdx).fault_class = classID;
        machines(machineIdx).total_lifecycle_days = randi([LC_MIN, LC_MAX]);

        % Assign unique shapes (alphas) for each potential fault
        machines(machineIdx).alpha_leak    = 1.5 + 1.5 * rand();
        machines(machineIdx).alpha_block   = 1.5 + 1.5 * rand();
        machines(machineIdx).alpha_bearing = 1.5 + 1.5 * rand();

        % Assign independent "Phantom Lifespan" multipliers
        % Multiplier > 1 ensures secondary faults don't reach 100% severity
        machines(machineIdx).mult_leak    = 1.0 + 0.7 * rand(); 
        machines(machineIdx).mult_block   = 1.0 + 0.7 * rand();
        machines(machineIdx).mult_bearing = 1.0 + 0.7 * rand();

        % Randomly select the "Dominant Fault" for this machine
        % (1=Leak, 2=Blockage, 3=Bearing)
        if classID == CLASS_ALL_FAULTS
            dom_f = randi([1, 3]); 
        elseif classID == CLASS_LEAK_BLOCK
            dom_f = randi([1, 2]);
        elseif classID == CLASS_BLOCK_BEARING
            dom_f = randi([2, 3]);
        elseif classID == CLASS_BEARING_LEAK
            dom_f = (rand() > 0.5) * 2 + 1; % Randomly returns 1 or 3
        end
        machines(machineIdx).dominant_fault = dom_f;

        % Force the dominant fault's multiplier to 1.0 (fails on exactly totalDays)
        if dom_f == 1,     machines(machineIdx).mult_leak = 1.0;
        elseif dom_f == 2, machines(machineIdx).mult_block = 1.0;
        elseif dom_f == 3, machines(machineIdx).mult_bearing = 1.0;
        end
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

% Tracks which machine each simulation belongs to
machine_tracker = zeros(1, totalSims);

% -----------------------------------------------------------------------
% SECTION 6 – Main Simulation Loop
% -----------------------------------------------------------------------
for k = 1:totalMachines

    mID = machines(k).machine_id;
    fClass = machines(k).fault_class;
    totalDays = machines(k).total_lifecycle_days;

    % Daily accumulators
    prev_t_leak  = 0;
    prev_t_block = 0;
    prev_t_bear  = 0;

    for day = 1:totalDays

        simCounter = simCounter + 1;

        machine_tracker(simCounter) = mID;

        % ================================================================
        % 1. LEAKAGE PROGRESSION
        % ================================================================
        t_eff_leak = 0;

        if ismember(fClass, ...
                [CLASS_LEAK_BLOCK, CLASS_BEARING_LEAK, CLASS_ALL_FAULTS])

            nt_p = (day - 1) / ...
                (totalDays * machines(k).mult_leak);

            nt_c = day / ...
                (totalDays * machines(k).mult_leak);

            inc = ...
                (nt_c ^ machines(k).alpha_leak) - ...
                (nt_p ^ machines(k).alpha_leak);

            t_eff_leak = min(1, ...
                prev_t_leak + inc * (0.8 + 0.4 * rand()));

            prev_t_leak = t_eff_leak;
        end

        % ================================================================
        % 2. BLOCKAGE PROGRESSION
        % ================================================================
        t_eff_block = 0;

        if ismember(fClass, ...
                [CLASS_LEAK_BLOCK, CLASS_BLOCK_BEARING, CLASS_ALL_FAULTS])

            nt_p = (day - 1) / ...
                (totalDays * machines(k).mult_block);

            nt_c = day / ...
                (totalDays * machines(k).mult_block);

            inc = ...
                (nt_c ^ machines(k).alpha_block) - ...
                (nt_p ^ machines(k).alpha_block);

            t_eff_block = min(1, ...
                prev_t_block + inc * (0.8 + 0.4 * rand()));

            prev_t_block = t_eff_block;
        end

        % ================================================================
        % 3. BEARING PROGRESSION
        % ================================================================
        t_eff_bear = 0;

        if ismember(fClass, ...
                [CLASS_BLOCK_BEARING, CLASS_BEARING_LEAK, CLASS_ALL_FAULTS])

            nt_p = (day - 1) / ...
                (totalDays * machines(k).mult_bearing);

            nt_c = day / ...
                (totalDays * machines(k).mult_bearing);

            inc = ...
                (nt_c ^ machines(k).alpha_bearing) - ...
                (nt_p ^ machines(k).alpha_bearing);

            t_eff_bear = min(1, ...
                prev_t_bear + inc * (0.8 + 0.4 * rand()));

            prev_t_bear = t_eff_bear;
        end

        % ================================================================
        % MAP TO REAL SEVERITY VALUES
        % ================================================================
        leak_sev = ...
            LEAK_MIN + ...
            (LEAK_MAX - LEAK_MIN) * t_eff_leak;

        block_sev = ...
            BLOCK_MAX - ...
            (BLOCK_MAX - BLOCK_MIN) * t_eff_block;

        bearing_sev = ...
            BEARING_MIN + ...
            (BEARING_MAX - BEARING_MIN) * t_eff_bear;

        % ================================================================
        % FINAL DAY OVERRIDE
        % ================================================================
        if day == totalDays

            if machines(k).dominant_fault == 1
                leak_sev = LEAK_MAX;

            elseif machines(k).dominant_fault == 2
                block_sev = BLOCK_MIN;

            elseif machines(k).dominant_fault == 3
                bearing_sev = BEARING_MAX;
            end
        end

        % ================================================================
        % CREATE SIMULATION INPUT
        % ================================================================
        si = Simulink.SimulationInput(mdl);

        si = setVariable(si, ...
            'leak_cyl_area_WKSP', leak_sev);

        si = setVariable(si, ...
            'block_in_factor_WKSP', block_sev);

        si = setVariable(si, ...
            'bearing_fault_frict_WKSP', bearing_sev);

        % Metadata
        si = setVariable(si, ...
            'machine_id_WKSP', mID);

        si = setVariable(si, ...
            'current_day_WKSP', day);

        si = setVariable(si, ...
            'fault_class_WKSP', fClass);

        si = setVariable(si, ...
            'total_lifecycle_days_WKSP', totalDays);

        % Store simulation input
        simInput(simCounter) = si;

    end
end
% -----------------------------------------------------------------------
% SECTION 7 – Split by Machine and Export
% -----------------------------------------------------------------------
fprintf('=== Splitting into 5 balanced chunks by expected simulation load ===\n');
team = {'Part1', 'Part2_RS_Raksha', 'Part3_Saanvi', 'Part4_RakshaS', 'Part5Gargi'};
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

    filename = sprintf('testing_%s_simInput.mat', team{i});
    save(filename, 'simInputBatch', 'chunkMeta', '-v7.3');
    fprintf('Saved %s (%d rows, machines: %s)\n', ...
        filename, chunkMeta.n_rows, mat2str(chunkMeta.machine_ids));
end
fprintf('\nDone! Send the .mat files to your team.\n');