function map = featureMapping(cfg)
%FEATUREMAPPING  Names, units and SHAP groupings for the 80 observation features.
%
%   MAP = FEATUREMAPPING(CFG) returns
%     .names       {1 x nObs} short machine names, e.g. 'V7', 'P_load24'
%     .labels      {1 x nObs} human labels used in plots and LLM prompts
%     .units       {1 x nObs}
%     .busOf       [1 x nObs] the bus a feature belongs to (NaN if none)
%     .groupIdx    [1 x nObs] index into the group list below
%     .groupNames  {1 x nGroup} human names of the SHAP groups
%     .groupMembers{1 x nGroup} feature indices in each group
%     .nGroup      number of groups
%
%   WHY GROUPS EXIST
%   ----------------
%   Exact Shapley values enumerate 2^M coalitions. With M = 80 raw
%   features that is 2^80 -- not large, but impossible. Two escape routes
%   exist and this project implements both:
%     (a) KernelSHAP on the raw 80 features (sampled coalitions), and
%     (b) EXACT Shapley values over a small number of semantic groups.
%   Route (b) is preferable for the LLM layer: "the tail-end voltages"
%   is a phrase an operator understands, whereas thirty-three individual
%   bus attributions are not something anyone reads. The grouping below
%   yields 12 groups, i.e. 4096 coalitions, which is exactly enumerable.
%
%   Grouping also fixes a real statistical problem. Adjacent bus voltages
%   on a radial feeder are almost perfectly correlated, and Shapley values
%   split credit arbitrarily among correlated features. Grouping the
%   correlated block and attributing to the block removes that ambiguity.
%
%   See also BUILDOBSERVATION, EXACTSHAP, KERNELSHAP.

if nargin < 1 || isempty(cfg)
    cfg = xrlConfig();
end

nObs = cfg.dims.nObs;
nPV  = cfg.dims.nPV;
nA   = cfg.dims.nAction;
nB   = cfg.dims.nBus;          % 33 on the reference feeder
oP   = 2*nB - 1;               % last load index (65 on the 33-bus feeder)

names  = cell(1, nObs);
labels = cell(1, nObs);
units  = cell(1, nObs);
busOf  = nan(1, nObs);

% --- 1:33 voltages ---------------------------------------------------
for b = 1:nB
    names{b}  = sprintf('V%d', b);
    labels{b} = sprintf('Bus %d voltage', b);
    units{b}  = 'p.u.';
    busOf(b)  = b;
end

% --- 34:65 active loads at buses 2..33 -------------------------------
for b = 2:nB
    i = nB - 1 + b;                 % 34 for b = 2 ... 65 for b = 33
    names{i}  = sprintf('P_load%d', b);
    labels{i} = sprintf('Bus %d active load', b);
    units{i}  = 'p.u.';
    busOf(i)  = b;
end

% --- 66:68 PV outputs -------------------------------------------------
for k = 1:nPV
    i = oP + k;
    names{i}  = sprintf('P_PV%d', cfg.pv.buses(k));
    labels{i} = sprintf('PV output at bus %d', cfg.pv.buses(k));
    units{i}  = 'p.u.';
    busOf(i)  = cfg.pv.buses(k);
end

% --- context ----------------------------------------------------------
i = oP + nPV;
names{i+1}  = 'LoadScale';  labels{i+1} = 'Aggregate load level';   units{i+1} = '-';
names{i+2}  = 'ToD_sin';    labels{i+2} = 'Time of day (sine)';     units{i+2} = '-';
names{i+3}  = 'ToD_cos';    labels{i+3} = 'Time of day (cosine)';   units{i+3} = '-';

% --- previous action --------------------------------------------------
actNames = actionChannelNames(cfg);
for k = 1:nA
    j = i + 3 + k;
    names{j}  = sprintf('prev_%s', actNames{k});
    labels{j} = sprintf('Previous %s', actNames{k});
    units{j}  = '-';
end

map.names  = names;
map.labels = labels;
map.units  = units;
map.busOf  = busOf;
map.actionNames = actNames;

% =====================================================================
% SHAP groups (12) -- semantic blocks an operator would name out loud
% =====================================================================
G = {};
netData = struct();
if nB ~= 33
    netData = cfg.net.dataFcn();
end
if isfield(netData, 'shapVoltageGroups')
    % Feeder-specific spatial bands (e.g. IEEE69BUSDATA). Bus lists map
    % to observation indices: voltage of bus b is index b, active load
    % of bus b is index nB-1+b.
    for g = 1:size(netData.shapVoltageGroups, 1)
        G{end+1} = struct('name', netData.shapVoltageGroups{g,1}, ...
                          'idx',  netData.shapVoltageGroups{g,2}); %#ok<AGROW>
    end
    for g = 1:size(netData.shapLoadGroups, 1)
        G{end+1} = struct('name', netData.shapLoadGroups{g,1}, ...
                          'idx',  nB - 1 + netData.shapLoadGroups{g,2}); %#ok<AGROW>
    end
else
    % IEEE 33-bus reference grouping (unchanged).
    G{end+1} = struct('name','Head-end voltages (bus 2-6)',      'idx', 2:6);
    G{end+1} = struct('name','Mid-feeder voltages (bus 7-13)',   'idx', 7:13);
    G{end+1} = struct('name','Feeder-end voltages (bus 14-18)',  'idx', 14:18);
    G{end+1} = struct('name','Lateral A voltages (bus 19-22)',   'idx', 19:22);
    G{end+1} = struct('name','Lateral B voltages (bus 23-25)',   'idx', 23:25);
    G{end+1} = struct('name','Lateral C voltages (bus 26-33)',   'idx', 26:33);
    G{end+1} = struct('name','Main-feeder load (bus 2-17)',      'idx', 34:49);
    G{end+1} = struct('name','End and lateral load (bus 18-33)', 'idx', 50:65);
end
G{end+1} = struct('name','PV active output',                 'idx', oP+(1:nPV));
G{end+1} = struct('name','Aggregate load level',             'idx', oP+nPV+1);
G{end+1} = struct('name','Time of day',                      'idx', oP+nPV+(2:3));
G{end+1} = struct('name','Previous control action',          'idx', (oP+nPV+4):nObs);

% Bus 1 voltage is the slack and is set by the OLTC, so it is excluded
% from the voltage groups -- attributing the agent's decision to a
% quantity the agent itself just set would be misleading.
groupIdx     = zeros(1, nObs);
groupNames   = cell(1, numel(G));
groupMembers = cell(1, numel(G));
for g = 1:numel(G)
    groupNames{g}   = G{g}.name;
    groupMembers{g} = G{g}.idx;
    groupIdx(G{g}.idx) = g;
end

map.groupIdx     = groupIdx;
map.groupNames   = groupNames;
map.groupMembers = groupMembers;
map.nGroup       = numel(G);

% ---------------------------------------------------------------------
% COARSE GROUPING (5 players) -- an alternative that trades spatial
% resolution for attribution reliability.
%
% The 12-group scheme above splits the feeder voltage into six spatial
% bands. On a radial feeder those bands are ~0.95-0.97 correlated, so
% Shapley credit for "the feeder is low" is divided arbitrarily among
% them and the RANKING of near-tied groups becomes noise -- measured
% faithfulness 0.306 +/- 0.467 on the 6000-episode agent. Collapsing the
% six voltage bands into a single "feeder voltage" player removes the
% split. In simulation this lifts ranking faithfulness from ~0.30 to
% ~0.50. The cost is that the explanation can no longer say WHICH end of
% the feeder drove the action, only that voltage did.

%
% Select with cfg.shap.grouping = 'coarse'.
Gc = {};
allV = 2:nB;                                   % every bus voltage except slack
Gc{end+1} = struct('name','Feeder voltage (all buses)', 'idx', allV);
Gc{end+1} = struct('name','Feeder load',               'idx', nB+1:oP);
Gc{end+1} = struct('name','PV active output',          'idx', oP+(1:nPV));
Gc{end+1} = struct('name','Time and load level',       'idx', [oP+nPV+1, oP+nPV+(2:3)]);
Gc{end+1} = struct('name','Previous control action',   'idx', (oP+nPV+4):nObs);

cGroupIdx = zeros(1,nObs); cNames = cell(1,numel(Gc)); cMem = cell(1,numel(Gc));
for g = 1:numel(Gc)
    cNames{g} = Gc{g}.name; cMem{g} = Gc{g}.idx; cGroupIdx(Gc{g}.idx) = g;
end
map.coarse.groupIdx     = cGroupIdx;
map.coarse.groupNames   = cNames;
map.coarse.groupMembers = cMem;
map.coarse.nGroup       = numel(Gc);

% Sanity: every feature except V1 must belong to exactly one group.
unassigned = find(groupIdx == 0);
assert(isequal(unassigned, 1), ...
    'featureMapping:coverage', ...
    'Only V1 (slack) should be ungrouped; found %s.', mat2str(unassigned));
end

% =====================================================================
function nm = actionChannelNames(cfg)
%ACTIONCHANNELNAMES  Human names for the 9 action channels.
nm = cell(1, cfg.dims.nAction);
nm{1} = 'OLTC tap';
for k = 1:cfg.dims.nCap
    nm{1+k} = sprintf('Cap bank bus %d', cfg.cap.buses(k));
end
for k = 1:cfg.dims.nPV
    nm{1+cfg.dims.nCap+k} = sprintf('PV inverter Q bus %d', cfg.pv.buses(k));
end
end
