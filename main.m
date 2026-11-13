function out = main(varargin)
%MAIN  Run the XRL-LLM Volt/VAR pipeline end to end.
%
%   OUT = MAIN()                       run everything with defaults
%   OUT = MAIN('stages', {'shap'})     run selected stages only
%   OUT = MAIN('policy', 'rule')       skip training, use the rule baseline
%   OUT = MAIN('quick', true)          small budgets, for a smoke test
%
%   STAGES (in dependency order)
%     'check'    dependency report
%     'test'     component test suite
%     'train'    PPO training                      (needs RL Toolbox)
%     'metrics'  closed-loop control performance vs. the rule baseline
%     'shap'     attribution records
%     'explain'  natural-language explanations
%     'evaluate' counterfactual faithfulness, G-Eval, ablations
%     'figures'  SHAP and voltage-profile plots
%     'dashboard' open the interactive console
%
%   TYPICAL FIRST RUN
%   -----------------
%       cd XRL_LLM_Project
%       setupPaths
%       main('quick', true, 'policy', 'rule')
%
%   That completes in a couple of minutes with no toolboxes beyond base
%   MATLAB and exercises every stage except training. Once it passes, run
%   MAIN() to train the agent and produce the full result set.
%
%   READING THE OUTPUT
%   ------------------
%   OUT.metrics.rl vs OUT.metrics.rule is the comparison that matters. The
%   uncontrolled column is printed as well, but see CONTROLMETRICS for why
%   a violation-reduction figure against the uncontrolled case does not
%   distinguish this work from conventional practice on this feeder.
%
%   See also SETUPPATHS, CHECKDEPENDENCIES, TRAINPPO, COLLECTATTRIBUTIONS.

root = setupPaths(); %#ok<NASGU>

p = inputParser;
p.addParameter('stages', {'check','train','metrics','shap','explain','evaluate','figures'});
p.addParameter('policy', 'auto');       % 'auto' | 'rule' | agent object
p.addParameter('quick',  false);
p.addParameter('cfg',    []);
p.parse(varargin{:});
A = p.Results;

cfg = A.cfg;
if isempty(cfg), cfg = xrlConfig(); end

if A.quick
    cfg.train.maxEpisodes    = 150;
    cfg.train.plots          = 'none';
    cfg.shap.nBackground     = 16;
    cfg.eval.nCounterfactual = 20;
    cfg.eval.nGEvalSamples   = 15;
    nShapEpisodes = 3;
    nMetricEpisodes = 15;
    maxRecords = 15;
else
    nShapEpisodes = 12;
    nMetricEpisodes = 60;
    maxRecords = cfg.shap.topK * 20;
end

stages = A.stages;
if ischar(stages) || isstring(stages), stages = cellstr(stages); end
has = @(s) any(strcmpi(stages, s));

rng(cfg.seed);
feeder = buildFeeder(cfg.net.dataFcn());
out = struct('cfg', cfg, 'feeder', feeder);

banner('XRL-LLM: Explainable Volt/VAR Control on the IEEE 33-bus feeder');

% =====================================================================
% Stage: dependency check
% =====================================================================
if has('check')
    out.deps = checkDependencies(true);
else
    out.deps = checkDependencies(false);
end

% =====================================================================
% Stage: component tests
% =====================================================================
if has('test')
    banner('Component tests');
    out.testsPassed = runAllTests();
end

% =====================================================================
% Stage: training
% =====================================================================
policySpec = A.policy;
out.agent = [];

if has('train') && ~strcmpi(policySpec,'rule')
    if out.deps.canTrain
        banner('PPO training');
        [out.agent, out.trainingStats] = trainPPO(cfg, feeder);
        policySpec = out.agent;
        try
            out.fig.training = plotTrainingProgress(out.trainingStats, cfg);
        catch
        end
    else
        warning('main:noTraining', ...
            ['Reinforcement Learning Toolbox is unavailable, so training is ', ...
             'skipped and the rule baseline is used instead. Every downstream ', ...
             'stage still runs.']);
        policySpec = 'rule';
    end
elseif strcmpi(policySpec,'auto')
    % No training requested: reload a saved agent if one exists.
    if exist(cfg.train.savePath,'file') && out.deps.canTrain
        S = load(cfg.train.savePath);
        out.agent = S.agent;
        policySpec = out.agent;
        fprintf('  Loaded a trained agent from %s\n', cfg.train.savePath);

        % Guard against train/serve mismatch. A reward-weight change is
        % silent: everything runs, and every reward-derived number
        % describes a reward this policy never optimised.
        if isfield(S,'cfg')
            [okCfg, cfgDiffs] = checkAgentConfig(S.cfg, cfg);
            out.configMatch  = okCfg;
            out.configDiffs  = cfgDiffs;
            if ~okCfg
                warning('main:configMismatch', ...
                    ['The reloaded agent was trained under a different ', ...
                     'configuration (%d field(s) differ). See the listing above. ', ...
                     'To reproduce its training-time numbers, use the saved ', ...
                     'config: S = load(cfg.train.savePath); cfg = S.cfg;'], ...
                     numel(cfgDiffs));
            end
        end
    else
        policySpec = 'rule';
    end
end

policyFcn = makePolicyFcn(policySpec, cfg);
out.policyIsRule = ischar(policySpec) || isstring(policySpec);
out.policyFcn = policyFcn;

% =====================================================================
% Stage: control metrics
% =====================================================================
if has('metrics')
    banner('Closed-loop control performance');

    rulePolicyFcn = makePolicyFcn('rule', cfg);
    out.metrics.rule = controlMetrics(rulePolicyFcn, cfg, feeder, nMetricEpisodes, ...
        struct('label','rule baseline','seed',cfg.seed+1000));

    if ~out.policyIsRule
        out.metrics.rl = controlMetrics(policyFcn, cfg, feeder, nMetricEpisodes, ...
            struct('label','PPO agent','seed',cfg.seed+1000));
        compareTable(out.metrics.rl, out.metrics.rule);
    else
        fprintf(['\n  Only the rule baseline was evaluated (no trained agent).\n' ...
                 '  Train an agent and re-run to produce the comparison.\n']);
    end
end

% =====================================================================
% Stage: SHAP attribution
% =====================================================================
if has('shap') || has('explain') || has('evaluate') || has('figures') || has('dashboard')
    banner('SHAP attribution');
    [out.records, out.shapMeta] = collectAttributions(policyFcn, cfg, feeder, ...
        nShapEpisodes, struct('maxRecords', maxRecords));
    fprintf('  %d decisions explained in %.1f s (%.2f s each, method: %s)\n', ...
        numel(out.records), out.shapMeta.elapsedSec, ...
        out.shapMeta.secPerRecord, out.shapMeta.method);

    if isempty(out.records)
        warning('main:noRecords', ...
            'No records produced; the remaining stages cannot run.');
        return
    end

    shares = arrayfun(@(r) r.topFeatures(1).share, out.records);
    fprintf('  Mean top-1 attribution share: %.3f (higher = more concentrated)\n', mean(shares));
end

% =====================================================================
% Stage: explanations
% =====================================================================
if has('explain') || has('evaluate') || has('dashboard')
    banner('Natural-language explanations');
    out.llm = connectLLM(cfg);
    fprintf('  Backend: %s (%s)\n', out.llm.backend, out.llm.model);

    n = numel(out.records);
    out.explanations = cell(n,1);
    lat = zeros(n,1);
    for i = 1:n
        [e, m] = generateLayeredExplanations(out.records(i), out.llm, cfg, feeder);
        out.explanations{i} = e;
        lat(i) = m.totalLatencySec;
    end
    out.meanLatencySec = mean(lat);
    fprintf('  Mean latency for all three levels: %.3f s (target < %.1f s) %s\n', ...
        out.meanLatencySec, cfg.eval.targetLatencySec, ...
        tern(out.meanLatencySec < cfg.eval.targetLatencySec,'[PASS]','[OVER TARGET]'));

    fprintf('\n  --- Example explanation (decision 1) ---\n');
    fprintf('  ACTION   : %s\n', out.records(1).actionText);
    fprintf('  SUMMARY  : %s\n', wrapText(out.explanations{1}.summary, 74, 13));
    fprintf('  STANDARD : %s\n', wrapText(out.explanations{1}.standard, 74, 13));

    stdExpl = cellfun(@(e) e.standard, out.explanations, 'UniformOutput', false);
    saveRecords(out.records, stdExpl, cfg, 'xrl');
end

% =====================================================================
% Stage: evaluation
% =====================================================================
if has('evaluate')
    banner('Counterfactual causal faithfulness');
    out.faithfulness = counterfactualTest(out.records, policyFcn, cfg, feeder);

    banner('Explanation quality');
    stdExpl = cellfun(@(e) e.standard, out.explanations, 'UniformOutput', false);
    out.quality = gEval(stdExpl, out.records, out.llm, cfg);

    banner('Ablation studies');
    out.ablation = ablationExperiments(out.records, out.llm, cfg, feeder, ...
        struct('nSamples', min(25, numel(out.records))));

    summariseTargets(out, cfg);
end

% =====================================================================
% Stage: figures
% =====================================================================
if has('figures')
    banner('Figures');
    phi = vertcat(out.records.phi);
    names = out.records(1).playerNames;

    out.fig.shapBar = plotSHAPBar(phi, names, ...
        'Global SHAP importance -- Volt/VAR control policy');

    vals = summariseGroupValues(out.records, out.shapMeta.groups);
    out.fig.shapBee = plotSHAPBeeswarm(phi, vals, names, ...
        'SHAP summary -- Volt/VAR control policy');

    out.fig.profile = plotVoltageProfile(out.records(1), cfg, feeder);

    fprintf('  Three figures created.\n');
end

% =====================================================================
% Stage: dashboard
% =====================================================================
if has('dashboard')
    banner('Dashboard');
    out.app = xrlDashboard(out.records, out.explanations, cfg, feeder);
end

banner('Done');
end

% =====================================================================
function banner(txt)
fprintf('\n%s\n', repmat('=',1,72));
fprintf('  %s\n', txt);
fprintf('%s\n', repmat('=',1,72));
end

function compareTable(rl, rule)
fprintf('\n  %-26s %14s %14s\n', 'metric', 'PPO agent', 'rule baseline');
fprintf('  %s\n', repmat('-',1,58));
fprintf('  %-26s %13.4f%% %13.4f%%\n', 'violated bus-steps', ...
    100*rl.violationRate, 100*rule.violationRate);
fprintf('  %-26s %14.1f %14.1f\n', 'mean loss (kW)', ...
    rl.meanLossKW, rule.meanLossKW);
fprintf('  %-26s %14.3f %14.3f\n', 'switching per step', ...
    rl.switchingPerStep, rule.switchingPerStep);
fprintf('  %-26s %14.3f %14.3f\n', 'interpretability penalty', ...
    rl.interpPenalty, rule.interpPenalty);
fprintf('  %s\n', repmat('-',1,58));
% SIGN CONVENTION: always report the CHANGE in the PPO agent relative to
% the rule baseline, so a negative number always means "less of this
% quantity". An earlier version reported a REDUCTION here (1 - a/b) while
% RUNFULLSTUDY reported a CHANGE ((a-b)/b) in its own table, so the two
% printed opposite signs for the same result. Quoting the wrong one would
% have reversed both headline claims in the paper. Never print a bare
% signed percentage without saying which direction is better.
dLoss = 100*(rl.meanLossKW       - rule.meanLossKW)       / max(abs(rule.meanLossKW),eps);
dSw   = 100*(rl.switchingPerStep - rule.switchingPerStep) / max(abs(rule.switchingPerStep),eps);
fprintf('  PPO relative to rule (negative = less):\n');
fprintf('    loss      %+.1f%%  (%s)\n', dLoss, betterWorse(dLoss));
fprintf('    switching %+.1f%%  (%s)\n', dSw,   betterWorse(dSw));
fprintf(['  Report THIS comparison. Violation reduction against the\n' ...
         '  uncontrolled feeder is saturated by conventional control here.\n']);
end

function summariseTargets(out, cfg)
fprintf('\n%s\n', repmat('=',1,72));
fprintf('  TARGETS FROM THE PROJECT BRIEF\n');
fprintf('%s\n', repmat('=',1,72));
fprintf('  %-28s %12s %12s   %s\n','metric','achieved','target','status');

if isfield(out,'metrics') && isfield(out.metrics,'rl')
    v = out.metrics.rl.violationReduction;
elseif isfield(out,'metrics')
    v = out.metrics.rule.violationReduction;
else
    v = NaN;
end
row('violation reduction', v, cfg.eval.targetViolRedux, '%.1f%%', 100);

if isfield(out,'quality')
    row('explanation quality', out.quality.meanOverall, cfg.eval.targetQuality, '%.2f', 1);
end
if isfield(out,'faithfulness')
    row('causal faithfulness', out.faithfulness.faithfulness, cfg.eval.targetFaithful, '%.3f', 1);
end
if isfield(out,'meanLatencySec')
    a = out.meanLatencySec; t = cfg.eval.targetLatencySec;
    fprintf('  %-28s %12.3f %12.1f   %s\n','inference latency (s)', a, t, ...
        tern(a < t,'PASS','OVER'));
end

if isfield(out,'quality') && out.quality.proxy
    fprintf(['\n  CAUTION: explanation quality was scored with the offline proxy\n' ...
             '  rubric, not an LLM judge. Do not report it as a G-Eval result.\n']);
end
fprintf('\n');

    function row(name, actual, target, fmt, scale)
        s = tern(actual >= target, 'PASS', 'BELOW');
        fprintf(['  %-28s ' fmt '%s ' fmt '%s   %s\n'], name, ...
            actual*scale, blanks(max(0,12-numel(sprintf(fmt,actual*scale)))), ...
            target*scale, blanks(max(0,12-numel(sprintf(fmt,target*scale)))), s);
    end
end

function s = betterWorse(d)
%BETTERWORSE  Spell out the direction so a signed percentage cannot be
%   misread. For both loss and switching, lower is better.
if d < 0
    s = sprintf('%.1f%% LOWER -- better', abs(d));
else
    s = sprintf('%.1f%% HIGHER -- worse', d);
end
end

function o = tern(c,a,b)
if c, o = a; else, o = b; end
end

function s = wrapText(txt, width, indent)
txt = char(txt);
words = strsplit(strtrim(txt));
lines = {}; cur = '';
for i = 1:numel(words)
    if isempty(cur)
        cur = words{i};
    elseif numel(cur) + 1 + numel(words{i}) <= width
        cur = [cur ' ' words{i}]; %#ok<AGROW>
    else
        lines{end+1} = cur; cur = words{i}; %#ok<AGROW>
    end
end
if ~isempty(cur), lines{end+1} = cur; end
s = strjoin(lines, [newline blanks(indent)]);
end
