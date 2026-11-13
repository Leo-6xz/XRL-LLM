%% make_fig4_profile.m
% Re-export Fig. 4 (voltage profile) at a midday decision of a high-PV or
% reverse-power day, instead of records(1), which fell at hour 0 (no PV).
%
% Uses the records already saved by Phase 2 (results\xrl_records.mat);
% nothing is retrained or re-explained. Run from the folder that contains
% XRL_LLM_Project.
%
% Why the old figure was misleading: the scenario tag ("reversePower",
% "highPV", ...) labels the whole simulated DAY, set by that day's PV peak
% versus load peak. At hour 0 PV output is zero, so the profile showed the
% night-time undervoltage side of a reverse-power day.

cd XRL_LLM_Project;
setupPaths;
%%
S      = load(fullfile('results','xrl_records.mat'), 'records', 'cfg');
recs   = S.records;
cfg    = mergeConfigDefaults(S.cfg);
feeder = buildFeeder(cfg.net.dataFcn());
base   = cfg.net.baseMVA;                       % p.u. -> MW

%% 1) Tabulate every explained decision ----------------------------------
n = numel(recs);
idx = (1:n)';
hour     = arrayfun(@(r) r.hour, recs(:));
scenario = arrayfun(@(r) string(r.scenario), recs(:));
pvMW     = arrayfun(@(r) sum(r.state.pvP)   * base, recs(:));
loadMW   = arrayfun(@(r) sum(r.state.Pload) * base, recs(:));
vMaxPre  = arrayfun(@(r) max(r.info.VmPre),  recs(:));
vMaxPost = arrayfun(@(r) max(r.info.VmPost), recs(:));
vMinPre  = arrayfun(@(r) min(r.info.VmPre),  recs(:));
vMinPost = arrayfun(@(r) min(r.info.VmPost), recs(:));
action   = arrayfun(@(r) string(r.actionBrief), recs(:));

T = table(idx, hour, scenario, pvMW, loadMW, vMaxPre, vMaxPost, ...
          vMinPre, vMinPost, action);

%% 2) Candidates: midday, PV-dominated days, PV actually producing ------
isMidday = hour >= 10 & hour <= 15;
isPVday  = ismember(scenario, ["reversePower","highPV"]);
cand = T(isMidday & isPVday & pvMW > 0.5, :);

if isempty(cand)
    warning(['No midday high-PV/reverse-power decision among the saved records. ' ...
             'Listing all decisions with PV > 0.5 MW instead.']);
    cand = T(pvMW > 0.5, :);
end

% Rank: reverse power flow first (PV > load), then the highest pre-action
% voltage, i.e. the decisions where the controller acted against
% overvoltage pressure. That is the situation the scenario label promises.
cand.reverseFlow = cand.pvMW > cand.loadMW;
cand = sortrows(cand, {'reverseFlow','vMaxPre'}, {'descend','descend'});

fprintf('\nTop candidates :\n');
disp(cand(1:min(10,height(cand)), :));

%% 3) Choose the record --------------------------------------------------
PICK = 4;          % <- override with any idx from the table above
r = recs(PICK);
fprintf('Using record %d: hour %d, %s day, PV %.2f MW, load %.2f MW\n', ...
        PICK, r.hour, r.scenario, sum(r.state.pvP)*base, sum(r.state.Pload)*base);

%% 4) Plot with your existing function, then replace the title -----------
fig = plotVoltageProfile(r, cfg, feeder);
ax  = findobj(fig, 'Type', 'axes');

dayLabel = containers.Map( ...
    {'reversePower','highPV','heavyLoad','lightLoad','nominal'}, ...
    {'reverse-power day','high-PV day','heavy-load day','light-load day','nominal day'});
dl = r.scenario;
if isKey(dayLabel, r.scenario), dl = dayLabel(r.scenario); end

% Report whichever limit is under pressure: V_max when PV dominates,
% V_min otherwise.
if max(r.info.VmPre) >= 1.0
    vTxt = sprintf('V_{max} %.4f \\rightarrow %.4f p.u.', max(r.info.VmPre), max(r.info.VmPost));
else
    vTxt = sprintf('V_{min} %.4f \\rightarrow %.4f p.u.', min(r.info.VmPre), min(r.info.VmPost));
end

SHOW_TITLE = true;   % IEEE figures often omit in-figure titles; set false
if SHOW_TITLE
    title(ax, {sprintf('Voltage profile at %02d:00, %s (PV %.2f MW, load %.2f MW)', ...
                       round(r.hour), dl, sum(r.state.pvP)*base, sum(r.state.Pload)*base), ...
               sprintf('Controller %s (%s)', r.actionBrief, vTxt)}, ...
          'FontSize', 10, 'FontWeight', 'normal');
else
    title(ax, '');
end
legend(ax, 'Location', 'best');

%% 5) Export at 600 dpi ----------------------------------------------------
outFile = fullfile(pwd, 'results', 'fig4_voltage_profile.png');
exportgraphics(fig, outFile, 'Resolution', 600);
fprintf('Saved %s\n', outFile);
