function study = runFullStudy(phase, cfg)
%RUNFULLSTUDY  Execute the complete XRL-LLM study in resource-safe phases.
%
%   study = runFullStudy('train')     % PPO training      -- Ollama STOPPED
%   study = runFullStudy('explain')   % SHAP + LLM + eval -- Ollama RUNNING
%   study = runFullStudy('repeats')   % ablation x5       -- overnight
%   study = runFullStudy('sweep')     % delta trade-off   -- overnight
%   study = runFullStudy('modelsize') % 3B vs 8B ablation -- Ollama RUNNING
%
%   WHY PHASES AND NOT ONE COMMAND
%   ------------------------------
%   MATLAB's Deep Learning Toolbox and Ollama both allocate GPU memory, and
%   Ollama decides how many layers to offload AT MODEL LOAD TIME. If MATLAB
%   is already holding VRAM when the model loads, Ollama silently falls back
%   to partial CPU inference -- a measured 58%/42% CPU/GPU split on a
%   12 GB card, which turned 3-5 s generations into 16 s. Nothing in either
%   tool reports this as an error; it shows up only as latency.
%
%   Running training and generation as separate phases, with Ollama stopped
%   for the first, avoids the contention entirely. It also means a crash in
%   the hour-long generation phase does not cost you the training run.
%
%   SEQUENCE
%   --------
%     1. Stop Ollama (system tray -> Quit).      runFullStudy('train')
%     2. Start Ollama.                           runFullStudy('explain')
%     3. Leave running overnight.                runFullStudy('repeats')
%                                                runFullStudy('sweep')
%
%   Every phase saves to results/ and can be re-run independently. Later
%   phases reload the trained agent from cfg.train.savePath automatically.
%
%   See also MAIN, TRAINPPO, ABLATIONREPEATS, REWARDWEIGHTSWEEP.

if nargin < 1 || isempty(phase), phase = 'train'; end
if nargin < 2 || isempty(cfg),   cfg   = xrlConfig(); end

setupPaths();
if ~exist(cfg.io.resultsDir,'dir'), mkdir(cfg.io.resultsDir); end

stamp = datestr(now,'yyyymmdd_HHMMSS');
study = struct('phase', phase, 'startedAt', datetime('now'), 'cfg', cfg);
tPhase = tic;

fprintf('\n%s\n  FULL STUDY -- phase: %s\n%s\n', repmat('=',1,72), upper(phase), repmat('=',1,72));

switch lower(phase)

% =====================================================================
case 'train'
% =====================================================================
    warnIfOllamaResident();
    reportGPU();

    fprintf('\n  Training %d episodes x %d steps. Expect 10-40 minutes.\n', ...
        cfg.train.maxEpisodes, cfg.episode.stepsPerEpisode);

    feeder = buildFeeder(cfg.net.dataFcn());
    [agent, stats, env] = trainPPO(cfg, feeder);   %#ok<ASGLU>

    study.agent  = agent;
    study.stats  = stats;
    study.feeder = feeder;

    try
        study.figTraining = plotTrainingProgress(stats, cfg);
        savefig(study.figTraining, fullfile(cfg.io.resultsDir, ...
            sprintf('training_curve_%s.fig', stamp)));
    catch ME
        warning('runFullStudy:noPlot','Could not plot training curve: %s', ME.message);
    end

    % Immediate head-to-head against the baseline, on identical scenarios.
    pf   = makePolicyFcn(agent, cfg);
    rule = makePolicyFcn('rule', cfg);
    study.metrics.rl   = controlMetrics(pf,   cfg, feeder, 60, ...
        struct('label','PPO agent','seed',cfg.seed+1000));
    study.metrics.rule = controlMetrics(rule, cfg, feeder, 60, ...
        struct('label','rule baseline','seed',cfg.seed+1000));

    printHeadToHead(study.metrics.rl, study.metrics.rule);

    saveable = stripHandles(study); %#ok<NASGU>
    save(fullfile(cfg.io.resultsDir, sprintf('phase_train_%s.mat', stamp)), ...
        '-struct','saveable','-v7.3');
    fprintf('\n  NEXT: start Ollama, then run  runFullStudy(''explain'')\n');

% =====================================================================
case 'explain'
% =====================================================================
    requireLLM(cfg);

    % main() reloads the saved agent when 'train' is absent from stages.
    study.out = main('stages', {'metrics','shap','explain','evaluate','figures'}, ...
                     'policy','auto', 'cfg', cfg);

    if study.out.policyIsRule
        warning('runFullStudy:noAgent', ...
            ['No trained agent was found at %s, so the rule baseline was used. ', ...
             'Run runFullStudy(''train'') first.'], cfg.train.savePath);
    end

    saveable = stripHandles(study); %#ok<NASGU>
    save(fullfile(cfg.io.resultsDir, sprintf('phase_explain_%s.mat', stamp)), ...
        '-struct','saveable','-v7.3');
    fprintf('\n  NEXT (overnight): runFullStudy(''repeats'') then runFullStudy(''sweep'')\n');

% =====================================================================
case 'repeats'
% =====================================================================
    requireLLM(cfg);

    % One deterministic record set, reused across repeats, so the only
    % thing varying is LLM generation -- which is the noise source being
    % measured. Regenerating records each repeat would confound the two.
    base = main('stages', {'shap'}, 'policy','auto', 'cfg', cfg);
    llm  = connectLLM(cfg);

    % Timing scales with the generator. Measured: ~16 s per generation on
    % llama3:8b at 58%/42% CPU/GPU offload; ~2 s on llama3.2:3b resident.
    nRec = min(50, numel(base.records));
    fprintf(['\n  5 repeats x 5 conditions x %d records (3 conditions call the LLM).\n' ...
             '  At ~16 s/generation (8B, partial offload) expect 6-8 hours;\n' ...
             '  at ~2 s (3B, resident) expect under an hour.\n' ...
             '  NOTE: generator (%s) and judge (%s) will not both fit in VRAM,\n' ...
             '  so Ollama reloads on every switch -- roughly 50 swaps this run.\n'], ...
        nRec, cfg.llm.model, cfg.llm.evalModel);

    study.repeats = ablationRepeats(base.records, llm, cfg, base.feeder, 5, ...
        struct('nSamples', min(50, numel(base.records))));
    study.records = base.records;

    saveable = stripHandles(study); %#ok<NASGU>
    save(fullfile(cfg.io.resultsDir, sprintf('phase_repeats_%s.mat', stamp)), ...
        '-struct','saveable','-v7.3');

% =====================================================================
case 'sweep'
% =====================================================================
    warnIfOllamaResident();

    feeder = buildFeeder(cfg.net.dataFcn());
    fprintf('\n  Trains one agent per grid point. Expect 2-6 hours.\n');

    study.sweep = rewardWeightSweep(cfg, feeder, struct( ...
        'alphaRange', 10, ...                       % hold alpha; delta is the story
        'deltaRange', [0 0.02 0.05 0.10 0.20], ...
        'episodes',   600, ...
        'evalEps',    25, ...
        'shapEps',    4));

    saveable = stripHandles(study); %#ok<NASGU>
    save(fullfile(cfg.io.resultsDir, sprintf('phase_sweep_%s.mat', stamp)), ...
        '-struct','saveable','-v7.3');
    fprintf(['\n  Plot sweep.lossKW against sweep.topShare with delta as the\n' ...
             '  parameter. A knee supports the penalty; a straight line means\n' ...
             '  it only trades one objective for another. Report either.\n']);

% =====================================================================
case 'modelsize'
% =====================================================================
    requireLLM(cfg);

    base = main('stages', {'shap'}, 'policy','auto', 'cfg', cfg);

    % Which models to compare. Override with cfg.llm.compareModels.
    % Both must already be pulled -- `ollama list` to check.
    if isfield(cfg.llm,'compareModels') && ~isempty(cfg.llm.compareModels)
        models = cfg.llm.compareModels;
    else
        models = {'llama3:8b','llama3.2:3b'};
    end

    avail = true(1,numel(models));
    for i = 1:numel(models)
        c = cfg; c.llm.model = models{i}; c.io.verbose = false;
        avail(i) = getfield(connectLLM(c), 'available'); %#ok<GFLD>
    end
    if nnz(avail) < 2
        error('runFullStudy:needTwoModels', ...
            ['The model-size ablation needs TWO reachable models; only %d of ', ...
             '{%s} responded. Pull the missing one (e.g. ollama pull llama3.2:3b) ', ...
             'or set cfg.llm.compareModels to two models you already have. ', ...
             'Failing now rather than spending an hour on a one-model result.'], ...
            nnz(avail), strjoin(models,', '));
    end
    % Accumulate into a CELL, not a struct array.
    %
    % The previous version pre-declared a 3-field empty struct array and
    % then assigned a 5-field element into it, which throws "Subscripted
    % assignment between dissimilar structures". The declaration and the
    % element had drifted apart across edits, and because the failure
    % happens AFTER ablationExperiments returns, it discarded ~30 minutes
    % of completed generation before anyone found out. A cell array cannot
    % drift, and the struct array is assembled once at the end from
    % whatever actually succeeded.
    msCell = {};
    for i = 1:numel(models)
        c = cfg; c.llm.model = models{i};
        fprintf('\n  --- model: %s ---\n', models{i});
        l = connectLLM(c);
        if ~l.available
            warning('runFullStudy:modelMissing', ...
                'Model %s unavailable; skipping. Pull it with: ollama pull %s', ...
                models{i}, models{i});
            continue
        end

        % Build fresh each iteration so a skipped model cannot leave stale
        % fields behind from the previous one.
        clear s
        s = struct();
        s.model       = models{i};
        s.sizeNote    = 'state parameter count and quantisation in the paper';
        s.nRecords    = numel(base.records);
        s.ablation    = ablationExperiments(base.records, l, c, base.feeder, ...
            struct('nSamples', min(50, numel(base.records))));
        % Latency of the GENERATING conditions only. Averaging across all
        % five includes shapOnly and template, which make no LLM call at
        % all (0.000 s), and therefore roughly halves the apparent figure:
        % 8B printed 7.88 s when the true per-explanation cost was 16.35 s,
        % and 3B printed 1.84 s against a true 3.26 s -- which made a model
        % that MISSES the 2 s interactive target look like it meets it.
        lat = [s.ablation.conditions.meanLatency];
        gen = lat > 0.01;                       % conditions that actually generate
        s.meanLatency    = mean(lat(gen));      % per-explanation generation cost
        s.meanLatencyAll = mean(lat);           % retained for comparability only
        s.fullLatency    = lat(strcmp({s.ablation.conditions.name},'full'));

        msCell{end+1} = s; %#ok<AGROW>

        % Checkpoint after EVERY model. Each one costs ~30 minutes of LLM
        % generation; a failure in the second must not destroy the first.
        try
            partial = struct('modelSize', {msCell}, 'models', {models}, ...
                             'completed', i, 'cfg', cfg);
            save(fullfile(cfg.io.resultsDir, ...
                sprintf('phase_modelsize_partial_%s.mat', stamp)), ...
                '-struct', 'partial', '-v7.3');
        catch ME
            warning('runFullStudy:partialSaveFailed', ...
                'Could not checkpoint after model %s: %s', models{i}, ME.message);
        end
    end

    % Assemble the struct array once, from whatever succeeded.
    if isempty(msCell)
        study.modelSize = struct('model',{},'sizeNote',{},'nRecords',{}, ...
                                 'ablation',{},'meanLatency',{});
    else
        study.modelSize = [msCell{:}];
    end

    if numel(study.modelSize) < 2
        warning('runFullStudy:modelsizeIncomplete', ...
            ['Only %d of %d models completed, so there is no comparison. ', ...
             'Completed results are still saved.'], numel(study.modelSize), numel(models));
    end
    if numel(study.modelSize) == 2
        a = study.modelSize(1); b = study.modelSize(2);
        ia = find(strcmp({a.ablation.conditions.name},'full'),1);
        ib = find(strcmp({b.ablation.conditions.name},'full'),1);
        fprintf('\n  --- Model size comparison (full condition) ---\n');
        fprintf('    %-14s quality %.2f   latency %.2f s/explanation\n', a.model, ...
            a.ablation.conditions(ia).meanOverall, a.fullLatency);
        fprintf('    %-14s quality %.2f   latency %.2f s/explanation\n', b.model, ...
            b.ablation.conditions(ib).meanOverall, b.fullLatency);
        fprintf('    speedup %.1fx for %.2f quality points\n', ...
            a.fullLatency/max(b.fullLatency,eps), ...
            a.ablation.conditions(ia).meanOverall - b.ablation.conditions(ib).meanOverall);

        % Prompt sensitivity: how much does the CONTENT of the prompt move
        % the score? A model that scores the same whatever it is told is
        % not using the domain context, however good its average looks.
        spreadA = range([a.ablation.conditions.meanOverall]);
        spreadB = range([b.ablation.conditions.meanOverall]);
        fprintf('    prompt sensitivity (quality spread across conditions):\n');
        fprintf('      %-14s %.2f\n', a.model, spreadA);
        fprintf('      %-14s %.2f\n', b.model, spreadB);
        if spreadB < 0.5*spreadA
            fprintf(['      -> the smaller model is largely INSENSITIVE to prompt\n' ...
                     '         content. Its explanations are cheaper but they are not\n' ...
                     '         responding to the attribution or the grid context.\n']);
        end
        fprintf(['    If quality is comparable, the DOMAIN PROMPT is doing the\n' ...
                 '    work rather than model scale -- report that explicitly.\n']);
    end

    saveable = stripHandles(study); %#ok<NASGU>
    save(fullfile(cfg.io.resultsDir, sprintf('phase_modelsize_%s.mat', stamp)), ...
        '-struct','saveable','-v7.3');

otherwise
    error('runFullStudy:badPhase', ...
        'Phase must be train, explain, repeats, sweep or modelsize.');
end

study.elapsedSec = toc(tPhase);
fprintf('\n  Phase "%s" finished in %.1f min.\n\n', phase, study.elapsedSec/60);
end

% =====================================================================
function t = stripHandles(s)
%STRIPHANDLES  Remove graphics handles before SAVE.
%   A Figure object serialised into a .mat file bloats it by megabytes and
%   deserialises as a dead handle, so it is worse than useless. Figures are
%   saved separately with SAVEFIG where they are wanted.
t = s;
fn = fieldnames(t);
for i = 1:numel(fn)
    v = t.(fn{i});
    if isa(v,'matlab.ui.Figure') || isa(v,'matlab.graphics.Graphics') || ...
       isa(v,'matlab.ui.control.UIControl')
        t = rmfield(t, fn{i});
    elseif isstruct(v) && isscalar(v)
        t.(fn{i}) = stripHandles(v);
    end
end
end

% =====================================================================
function requireLLM(cfg)
l = connectLLM(cfg);
if ~l.available
    error('runFullStudy:noLLM', ...
        ['This phase needs a live LLM. Start Ollama and confirm with ', ...
         '`ollama ps`. connectLLM reported: %s'], l.lastError);
end
if strcmp(cfg.llm.model, cfg.llm.evalModel)
    warning('runFullStudy:selfJudge', ...
        ['Generator and judge are both %s. Scores will be inflated by ', ...
         'self-preference. Set cfg.llm.evalModel to a different model.'], cfg.llm.model);
end
end

% =====================================================================
function warnIfOllamaResident()
%WARNIFOLLAMARESIDENT  Nudge the user to free VRAM before a training phase.
try
    c = ollamaChat('llama3:8b','TimeOut',3);
    generate(c,'OK');
    warning('runFullStudy:ollamaRunning', ...
        ['Ollama appears to be running and holding GPU memory. Training will ', ...
         'compete with it for VRAM. Quit Ollama from the system tray before ', ...
         'this phase, then restart it for the explain phase.']);
catch
    % Not reachable, which is what we want here.
end
end

% =====================================================================
function reportGPU()
try
    g = gpuDevice;
    fprintf('  GPU: %s, %.1f GB free of %.1f GB\n', g.Name, ...
        g.AvailableMemory/1e9, g.TotalMemory/1e9);
catch
    fprintf('  GPU: not available to MATLAB (training will use the CPU).\n');
end
end

% =====================================================================
function printHeadToHead(rl, rule)
fprintf('\n  %-26s %14s %14s %10s\n','metric','PPO agent','rule baseline','delta');
fprintf('  %s\n', repmat('-',1,68));
row('violated bus-steps (%%)', 100*rl.violationRate, 100*rule.violationRate);
row('mean loss (kW)',          rl.meanLossKW,        rule.meanLossKW);
row('switching per step',      rl.switchingPerStep,  rule.switchingPerStep);
row('interpretability penalty',rl.interpPenalty,     rule.interpPenalty);
fprintf('  %s\n', repmat('-',1,68));
fprintf(['  Loss and switching are the reportable claims. Violation reduction\n' ...
         '  is saturated by conventional control on this feeder.\n']);
    function row(name, a, b)
        % Change of PPO relative to rule. Negative = less of this
        % quantity = better for every metric in this table.
        if abs(b) > eps, d = 100*(a-b)/abs(b); else, d = NaN; end
        fprintf('  %-26s %14.3f %14.3f %9.1f%%\n', name, a, b, d);
    end
end
