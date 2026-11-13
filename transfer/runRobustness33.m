function R = runRobustness33(opts)
%RUNROBUSTNESS33  Robustness of the TRAINED 33-bus agent and its explanations.
%
%   R = RUNROBUSTNESS33() evaluates the agent saved by runFullStudy('train')
%   under conditions it never saw in training, WITHOUT retraining, and with
%   the conventional (rule) baseline subjected to the identical conditions
%   on the identical scenarios (same seed as Table II, cfg.seed + 1000).
%   No language model is needed; Ollama can stay stopped. ~10-20 min.
%
%   Control robustness (60 episodes per condition, RL and rule)
%   ------------------------------------------------------------
%     C0  nominal                              (must reproduce Table II)
%     C1  voltage telemetry noise, sigma = 0.005 p.u. (0.5%)
%     C2  voltage telemetry noise, sigma = 0.010 p.u. (1.0%)
%     C3  load and PV telemetry error, 10% multiplicative (sigma = 0.10)
%     C4  out-of-range loading: daily peak 1.30-1.50 (training 0.60-1.30)
%     C5  PV capacity +50%: 3 x 1800 kW (training 3 x 1200 kW)
%     C6  capacitor bank at bus 30 out of service (stuck open)
%     C7  PV inverter at bus 32 out of service (unity power factor)
%     C8  PV relocated to buses 18, 22, 33 (zero-shot configuration change)
%   Telemetry noise corrupts only what the controller SEES; the physics,
%   the reward and the violation count use the true state.
%
%   Explanation robustness (attribution stability)
%   ----------------------------------------------
%   For the explained decisions of Phase 2 (same seed, same 12 episodes,
%   same background), the exact grouped Shapley vector is recomputed with
%   the explained observation corrupted by voltage telemetry noise
%   (sigma = 0.005 and 0.010 p.u., OPTS.nDraws draws each) and compared
%   with the clean attribution:
%     * Spearman rank correlation between clean and noisy attributions
%     * top-1 retention: share of draws whose leading group is unchanged
%     * top-3 overlap: |top3(clean) & top3(noisy)| / 3
%     * |delta a|: change in the explained setpoint itself
%   This asks whether an operator would be told a different story because
%   of ordinary measurement error, which is the explanation-side analogue
%   of the control robustness tests above.
%
%   OPTS fields (all optional)
%     .nEpisodes   episodes per condition           (default 60)
%     .nDraws      noise draws per explained record (default 3)
%     .maxRecords  explained records used           (default 100)
%     .skipControl / .skipExplain   run only one half
%
%   Output: struct R, saved to results/robustness33.mat; a formatted
%   summary is printed and written to results/robustness33_summary.txt.
%
%   See also CONTROLMETRICS, COLLECTATTRIBUTIONS, RUNTRANSFER69.

if nargin < 1, opts = struct(); end
opts = setDefault(opts, 'nEpisodes', 60);
opts = setDefault(opts, 'nDraws', 3);
opts = setDefault(opts, 'maxRecords', 100);
opts = setDefault(opts, 'skipControl', false);
opts = setDefault(opts, 'skipExplain', false);

root = setupPaths();
oldDir = cd(root);                       % all paths below are relative to the project root
restoreDir = onCleanup(@() cd(oldDir));
base = xrlConfig();
S = load(base.train.savePath);          % the agent reported in the paper
cfg0 = mergeConfigDefaults(S.cfg);      % its own training configuration
cfg0.io.verbose = false;
agent = S.agent;
nB = cfg0.dims.nBus;
seedEval = cfg0.seed + 1000;            % identical to Table II

cfg0.io.resultsDir = 'results';         % write next to the 33-bus results
if ~exist(cfg0.io.resultsDir, 'dir'), mkdir(cfg0.io.resultsDir); end
logFile = fullfile(root, cfg0.io.resultsDir, 'robustness33_summary.txt');
if exist(logFile, 'file'), delete(logFile); end
try
    diary(logFile);
catch ME
    warning('runRobustness33:diary', 'No log file (%s); copy the printed tables instead.', ME.message);
end
cleanup = onCleanup(@() diary('off'));

R = struct('opts', opts, 'startedAt', datetime('now'));

% =====================================================================
% 1) Control robustness
% =====================================================================
if ~opts.skipControl
    C = conditionList(cfg0, nB);
    rows = struct('name',{},'rl',{},'rule',{});
    fprintf('\n=== Control robustness (%d episodes per condition, seed %d) ===\n', ...
        opts.nEpisodes, seedEval);
    for i = 1:numel(C)
        c = C(i);
        cfg = c.cfgFcn(cfg0);
        feeder = buildFeeder(cfg.net.dataFcn());
        % The agent's observation normalisation was fixed at training time,
        % so it is NOT re-derived for C5/C8 -- that is what makes them
        % out-of-distribution tests.
        pfRL   = makePolicyFcn(agent, cfg);
        pfRule = makePolicyFcn('rule', cfg);

        rng(seedEval + i, 'twister');   % noise draws reproducible per condition
        rl = controlMetrics(c.wrap(pfRL),   cfg, feeder, opts.nEpisodes, ...
            struct('label', ['PPO / ' c.name], 'seed', seedEval, 'verbose', false));
        rng(seedEval + i, 'twister');
        ru = controlMetrics(c.wrap(pfRule), cfg, feeder, opts.nEpisodes, ...
            struct('label', ['rule / ' c.name], 'seed', seedEval, 'verbose', false));
        rows(end+1) = struct('name', c.name, 'rl', rl, 'rule', ru); %#ok<AGROW>
        fprintf('  %-40s done\n', c.name);
    end
    R.control = rows;
    printControlTable(rows);
end

% =====================================================================
% 2) Explanation robustness
% =====================================================================
if ~opts.skipExplain
    cfg = cfg0;
    feeder = buildFeeder(cfg.net.dataFcn());
    pf = makePolicyFcn(agent, cfg);
    fprintf('\n=== Attribution stability under voltage telemetry noise ===\n');
    fprintf('  Recomputing the Phase 2 records (12 episodes, seed %d) ...\n', cfg.seed);
    [recs, meta] = collectAttributions(pf, cfg, feeder, 12, ...
        struct('maxRecords', opts.maxRecords, 'verbose', false));
    map = featureMapping(cfg);
    [groups, ~] = shapGroups(cfg, map);
    sopts = struct('maxExactM', cfg.shap.maxExactM, 'batchSize', cfg.shap.batchSize, 'verbose', false);
    sigmas = [0.005 0.010];
    rs = RandStream('mt19937ar', 'Seed', seedEval);
    E = struct('sigma',{},'rho',{},'top1',{},'top3',{},'dAction',{},'scenario',{});
    for s = sigmas
        rho = []; top1 = []; top3 = []; dA = []; scen = {};
        for n = 1:numel(recs)
            r = recs(n);
            k = r.shapMeta.explainedChannel;
            predictFcn = shapValueFunction(pf, k, cfg);
            x = r.obs(:)';
            phi = r.phi(:)';
            [~, t1] = max(abs(phi));
            [~, o3] = sort(abs(phi), 'descend'); t3 = o3(1:3);
            for d = 1:opts.nDraws
                xn = x;
                xn(1:nB) = xn(1:nB) + s * randn(rs, 1, nB);
                phin = exactSHAP(predictFcn, xn, meta.background, groups, sopts);
                phin = phin(:)';
                [~, t1n] = max(abs(phin));
                [~, o3n] = sort(abs(phin), 'descend');
                rho(end+1)  = spearmanLocal(phi, phin);              %#ok<AGROW>
                top1(end+1) = double(t1n == t1);                     %#ok<AGROW>
                top3(end+1) = numel(intersect(t3, o3n(1:3))) / 3;    %#ok<AGROW>
                dA(end+1)   = abs(predictFcn(xn) - predictFcn(x));   %#ok<AGROW>
                scen{end+1} = r.scenario;                            %#ok<AGROW>
            end
        end
        E(end+1) = struct('sigma', s, 'rho', rho, 'top1', top1, 'top3', top3, ...
                          'dAction', dA, 'scenario', {scen}); %#ok<AGROW>
    end
    R.explain = E;
    R.nRecords = numel(recs);
    printExplainTable(E, numel(recs), opts.nDraws);
end

R.finishedAt = datetime('now');
save(fullfile(cfg0.io.resultsDir, 'robustness33.mat'), 'R');
fprintf('\nSaved results/robustness33.mat and %s\n', logFile);
end

% =====================================================================
function C = conditionList(cfg0, nB)
id = @(p) p;
C = struct('name',{},'cfgFcn',{},'wrap',{});
C(end+1) = struct('name','C0 Nominal (reproduces Table II)', 'cfgFcn', @(c) c, 'wrap', id);
C(end+1) = struct('name','C1 Voltage telemetry noise 0.5%', 'cfgFcn', @(c) c, ...
    'wrap', @(p) @(X) p(addNoise(X, 1:nB, 0.005, 'add')));
C(end+1) = struct('name','C2 Voltage telemetry noise 1.0%', 'cfgFcn', @(c) c, ...
    'wrap', @(p) @(X) p(addNoise(X, 1:nB, 0.010, 'add')));
C(end+1) = struct('name','C3 Load/PV telemetry error 10%', 'cfgFcn', @(c) c, ...
    'wrap', @(p) @(X) p(addNoise(X, (nB+1):(2*nB-1+cfg0.dims.nPV), 0.10, 'mult')));
C(end+1) = struct('name','C4 Load peak 1.30-1.50 (out of range)', ...
    'cfgFcn', @(c) setField(c, 'episode', 'loadScaleRange', [1.30 1.50]), 'wrap', id);
C(end+1) = struct('name','C5 PV capacity +50% (3 x 1800 kW)', ...
    'cfgFcn', @(c) setField(c, 'pv', 'ratedKW', 1.5*c.pv.ratedKW), 'wrap', id);
kCap = find(cfg0.cap.buses == 30, 1);
C(end+1) = struct('name','C6 Capacitor at bus 30 out of service', 'cfgFcn', @(c) c, ...
    'wrap', @(p) @(X) forceChannel(p(X), 1 + kCap, -1));
kPV = find(cfg0.pv.buses == 32, 1);
C(end+1) = struct('name','C7 PV inverter at bus 32 out of service', 'cfgFcn', @(c) c, ...
    'wrap', @(p) @(X) forceChannel(p(X), 1 + cfg0.dims.nCap + kPV, 0));
C(end+1) = struct('name','C8 PV relocated to buses 18, 22, 33', ...
    'cfgFcn', @(c) setField(c, 'pv', 'buses', [18 22 33]), 'wrap', id);
end

function X = addNoise(X, idx, sigma, mode)
switch mode
    case 'add',  X(:, idx) = X(:, idx) + sigma * randn(size(X,1), numel(idx));
    case 'mult', X(:, idx) = X(:, idx) .* max(0, 1 + sigma * randn(size(X,1), numel(idx)));
end
end

function A = forceChannel(A, k, v)
A(:, k) = v;
end

function c = setField(c, a, b, v)
c.(a).(b) = v;
end

function o = setDefault(o, f, v)
if ~isfield(o, f), o.(f) = v; end
end

function r = spearmanLocal(x, y)
rx = tiedRankLocal(x(:)); ry = tiedRankLocal(y(:));
rx = rx - mean(rx); ry = ry - mean(ry);
den = sqrt(sum(rx.^2) * sum(ry.^2));
if den == 0, r = NaN; else, r = sum(rx .* ry) / den; end
end

function r = tiedRankLocal(x)
[~, ord] = sort(x);
r = zeros(size(x)); r(ord) = 1:numel(x);
ux = unique(x);
for u = ux(:)'
    m = x == u;
    if nnz(m) > 1, r(m) = mean(r(m)); end
end
end

function printControlTable(rows)
fprintf('\n%-40s | %8s %8s | %7s %7s | %6s %6s | %6s\n', 'Condition', ...
    'Viol RL', 'Viol CV', 'Loss RL', 'Loss CV', 'Sw RL', 'Sw CV', 'OV RL');
fprintf('%s\n', repmat('-', 1, 104));
for i = 1:numel(rows)
    a = rows(i).rl; b = rows(i).rule;
    fprintf('%-40s | %7.3f%% %7.3f%% | %7.1f %7.1f | %6.3f %6.3f | %5.0f%%\n', rows(i).name, ...
        100*a.violationRate, 100*b.violationRate, a.meanLossKW, b.meanLossKW, ...
        a.switchingPerStep, b.switchingPerStep, 100*a.overvoltageShare);
end
fprintf('%s\n', repmat('-', 1, 104));
fprintf('Viol = violated bus-steps; Loss = mean active loss (kW); Sw = switching per step;\n');
fprintf('OV = share of RL violations that are overvoltage. CV = conventional (rule) baseline.\n');
fprintf('\nPer-regime violations (RL / CV):\n');
for i = 1:numel(rows)
    s = sprintf('  %-40s', rows(i).name);
    for j = 1:numel(rows(i).rl.byScenario)
        t = rows(i).rl.byScenario(j).tag;
        jr = find(strcmp({rows(i).rule.byScenario.tag}, t), 1);
        vr = NaN; if ~isempty(jr), vr = rows(i).rule.byScenario(jr).violationRate; end
        s = [s sprintf(' %s %.2f/%.2f', t, 100*rows(i).rl.byScenario(j).violationRate, 100*vr)]; %#ok<AGROW>
    end
    fprintf('%s\n', s);
end
end

function printExplainTable(E, nRec, nDraws)
fprintf('\n%d explained decisions x %d draws per noise level\n', nRec, nDraws);
fprintf('%-10s | %14s | %8s | %8s | %10s\n', 'sigma', 'Spearman', 'Top-1', 'Top-3', '|delta a|');
fprintf('%s\n', repmat('-', 1, 62));
for e = E
    fprintf('%7.3f    | %6.3f +/- %4.2f | %8.3f | %8.3f | %10.4f\n', e.sigma, ...
        mean(e.rho,'omitnan'), std(e.rho,'omitnan'), mean(e.top1), mean(e.top3), mean(e.dAction));
end
fprintf('%s\n', repmat('-', 1, 62));
fprintf('By regime (Spearman, sigma = %.3f):\n', E(end).sigma);
tags = unique(E(end).scenario);
for t = tags(:)'
    m = strcmp(E(end).scenario, t{1});
    fprintf('  %-14s n=%3d  rho %.3f  top-1 %.3f\n', t{1}, nnz(m), ...
        mean(E(end).rho(m),'omitnan'), mean(E(end).top1(m)));
end
end
