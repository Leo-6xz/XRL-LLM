%% relabel_load_groups.m
% Re-export Figs. 6 and 7 with load-group labels that match the actual
% observation indices. NOTHING is recomputed: the Shapley values, records
% and every number in the paper are unchanged; only two text labels change.
%
% Why: featureMapping assigns observation indices 34:49 to "Main-feeder
% load" and 50:65 to "Lateral load". Index i holds the load of bus i-32, so
% those blocks are buses 2-17 and 18-33. The old labels said 2-18 and 19-33;
% bus 18's load (90 kW, 2.4% of feeder demand) is actually in the second
% block.
%
% Run from XRL_LLM_Project (or its parent). Close any viewer/Word document
% that has the old PNGs open first, or the export will fail.

if ~exist(fullfile(pwd,'setupPaths.m'),'file') && exist('XRL_LLM_Project','dir')
    cd XRL_LLM_Project;
end
setupPaths;

OLD = {'Main-feeder load (bus 2-18)', 'Lateral load (bus 19-33)'};
NEW = {'Main-feeder load (bus 2-17)', 'End and lateral load (bus 18-33)'};

S    = load(fullfile('results','xrl_records.mat'), 'records', 'cfg');
recs = S.records;
cfg  = mergeConfigDefaults(S.cfg);

% Same inputs as main.m's figure stage
phi   = vertcat(recs.phi);
names = recs(1).playerNames;
for k = 1:numel(OLD)
    names = strrep(names, OLD{k}, NEW{k});
end
map    = featureMapping(cfg);
groups = shapGroups(cfg, map);
vals   = summariseGroupValues(recs, groups);

% Guard: the relabelled groups must be exactly the indices claimed
iMain = find(strcmp(names, NEW{1}));
iLat  = find(strcmp(names, NEW{2}));
assert(~isempty(iMain) && isequal(groups{iMain}(:)', 34:49), 'Main-feeder load group is not indices 34:49.');
assert(~isempty(iLat)  && isequal(groups{iLat}(:)',  50:65), 'Lateral load group is not indices 50:65.');

f6 = plotSHAPBar(phi, names, ' ', 'Mean SHAP value (impact on normalized setpoint)');
title(findobj(f6,'Type','axes'), '');
exportgraphics(f6, fullfile('results','fig6_shap_bar.png'), 'Resolution', 600);

f7 = plotSHAPBeeswarm(phi, vals, names, ' ');
title(findobj(f7,'Type','axes'), '');
exportgraphics(f7, fullfile('results','fig7_shap_beeswarm.png'), 'Resolution', 600);

fprintf('Saved results\\fig6_shap_bar.png and results\\fig7_shap_beeswarm.png\n');
