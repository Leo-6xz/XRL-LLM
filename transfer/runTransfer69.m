function T = runTransfer69(stages)
%RUNTRANSFER69  Transfer the XRL-LLM pipeline to the IEEE 69-bus feeder.
%
%   T = RUNTRANSFER69() runs, with the 33-bus hyperparameters unchanged
%   (see XRLCONFIG69):
%     'train'    PPO training, 6000 episodes          (Ollama STOPPED)
%     'metrics'  60 held-out episodes, PPO vs conventional vs uncontrolled,
%                same protocol and seed offset as Table II
%     'shap'     exact grouped Shapley attribution, fine (12) and coarse (5)
%     'faith'    counterfactual faithfulness, fine and coarse (Table IV protocol)
%     'llm'      OPTIONAL: three-level explanations for 5 decisions and live
%                latency (needs Ollama running; not in the default list)
%
%   T = RUNTRANSFER69({'metrics','shap','faith'}) skips training and reloads
%   results/ieee69/trained_ppo_agent_69.mat.
%
%   Run CHECKTRANSFER69 first. Everything is written to results/ieee69/,
%   so nothing from the 33-bus study is overwritten. The printed summary
%   (also in results/ieee69/transfer69_summary.txt) has the same rows as
%   Tables II-IV of our paper.
%
%   See also XRLCONFIG69, CHECKTRANSFER69, RUNROBUSTNESS33.

if nargin < 1 || isempty(stages), stages = {'train','metrics','shap','faith'}; end
has = @(s) any(strcmpi(stages, s));

root = setupPaths();
oldDir = cd(root);                       % all paths below are relative to the project root
restoreDir = onCleanup(@() cd(oldDir));
cfg = xrlConfig69();
if ~exist(cfg.io.resultsDir, 'dir'), mkdir(cfg.io.resultsDir); end
feeder = buildFeeder(cfg.net.dataFcn());

logFile = fullfile(root, cfg.io.resultsDir, 'transfer69_summary.txt');
try
    diary(logFile);
catch ME
    warning('runTransfer69:diary', 'No log file (%s); copy the printed tables instead.', ME.message);
end
cleanup = onCleanup(@() diary('off'));
fprintf('\n=== XRL-LLM transfer: IEEE 69-bus (%s) ===\n', datestr(now));
fprintf('  Observation %d-D, action %d-D, %d fine / %d coarse SHAP players\n', ...
    cfg.dims.nObs, cfg.dims.nAction, 12, 5);
fprintf('  Capacitors at %s, PV at %s\n', mat2str(cfg.cap.buses), mat2str(cfg.pv.buses));

T = struct('cfg', cfg, 'startedAt', datetime('now'));

% ------------------------------------------------------------------ train
if has('train')
    tic;
    [agent, stats] = trainPPO(cfg, feeder);
    T.trainSec = toc;
    S = load(cfg.train.savePath);
    T.bestEpisode = S.bestEpisode; T.bestAvgReward = S.bestAvgReward;
    T.finalAvgReward = S.finalAvgReward;
    try
        fig = plotTrainingProgress(stats, cfg);
        exportgraphics(fig, fullfile(cfg.io.resultsDir, 'training_curve_69.png'), 'Resolution', 300);
    catch
    end
else
    S = load(cfg.train.savePath);
    agent = S.agent;
    T.trainSec = S.elapsedSec;
    T.bestEpisode = S.bestEpisode; T.bestAvgReward = S.bestAvgReward;
    T.finalAvgReward = S.finalAvgReward;
end
fprintf('\n  Training: %.1f min; best 50-ep avg %.3f at ep %d; final %.3f\n', ...
    T.trainSec/60, T.bestAvgReward, T.bestEpisode, T.finalAvgReward);

pf   = makePolicyFcn(agent, cfg);
rule = makePolicyFcn('rule', cfg);

% ---------------------------------------------------------------- metrics
if has('metrics')
    T.rl   = controlMetrics(pf,   cfg, feeder, 60, struct('label','PPO (69-bus)','seed',cfg.seed+1000,'verbose',false));
    T.rule = controlMetrics(rule, cfg, feeder, 60, struct('label','rule (69-bus)','seed',cfg.seed+1000,'verbose',false));
    a = T.rl; b = T.rule;
    fprintf('\n  CONTROL, 60 held-out episodes          XRL-LLM   Conventional  Uncontrolled\n');
    fprintf('  Violated bus-steps (%%)               %8.3f   %10.3f   %10.3f\n', 100*a.violationRate, 100*b.violationRate, 100*a.violationRateUnc);
    fprintf('  Mean active loss (kW)                %8.1f   %10.1f   %10.1f\n', a.meanLossKW, b.meanLossKW, a.meanLossKWUnc);
    fprintf('  Switching per step                   %8.3f   %10.3f\n', a.switchingPerStep, b.switchingPerStep);
    fprintf('  Interpretability penalty             %8.3f   %10.3f\n', a.interpPenalty, b.interpPenalty);
    fprintf('  Loss vs conventional        %+6.1f%%\n', 100*(a.meanLossKW/b.meanLossKW - 1));
    fprintf('  Switching vs conventional   %+6.1f%%\n', 100*(a.switchingPerStep/b.switchingPerStep - 1));
    fprintf('  Penalty vs conventional     %+6.1f%%\n', 100*(a.interpPenalty/b.interpPenalty - 1));
    fprintf('  Overvoltage share of RL violations: %.0f%%\n', 100*a.overvoltageShare);
    fprintf('\n  By regime (violated bus-steps %%): RL / conventional / uncontrolled\n');
    for j = 1:numel(a.byScenario)
        t = a.byScenario(j).tag;
        jr = find(strcmp({b.byScenario.tag}, t), 1);
        fprintf('    %-14s n=%2d  %7.3f / %7.3f / %7.3f\n', t, a.byScenario(j).n, ...
            100*a.byScenario(j).violationRate, 100*b.byScenario(jr).violationRate, ...
            100*a.byScenario(j).violationRateUnc);
    end
end

% ------------------------------------------------------------ shap/faith
if has('shap') || has('faith')
    for scheme = {'fine','coarse'}
        c = cfg; c.shap.grouping = scheme{1};
        tic;
        [recs, meta] = collectAttributions(pf, c, feeder, 12, struct('maxRecords', 100, 'verbose', false));
        conc = mean(arrayfun(@(r) r.topFeatures(1).share, recs));
        effGap = max(arrayfun(@(r) r.shapMeta.efficiencyGap, recs));
        A = struct('records', recs, 'meta', meta, 'concentration', conc, ...
                   'efficiencyGap', effGap, 'secPerDecision', meta.secPerRecord);
        fprintf('\n  SHAP (%s, %d players): %d decisions, %.2f s/decision, max efficiency gap %.1e, concentration %.3f\n', ...
            scheme{1}, numel(recs(1).playerNames), numel(recs), meta.secPerRecord, effGap, conc);
        if has('faith')
            F = counterfactualTest(recs, pf, c, feeder);
            A.faith = F;
            fprintf('    Faithfulness: rank corr %.3f +/- %.3f | top-1 %.3f | sign %.3f\n', ...
                F.faithfulness, F.faithfulnessSD, F.top1Rate, F.signAgreement);
            for j = 1:numel(F.byScenario)
                fprintf('      %-14s n=%2d  %.3f\n', F.byScenario(j).tag, F.byScenario(j).n, F.byScenario(j).faithfulness);
            end
        end
        % Leading groups, to compare the spatial story with the 33-bus feeder
        names = recs(1).playerNames;
        phi = vertcat(recs.phi);
        [~, ord] = sort(mean(abs(phi), 1), 'descend');
        fprintf('    Top groups by mean |phi|: %s\n', strjoin(names(ord(1:min(3,end))), '; '));
        T.(scheme{1}) = A;
    end
    saveRecords(T.fine.records, {}, cfg, 'xrl69');
end

% -------------------------------------------------------------- optional LLM
if has('llm')
    llm = connectLLM(cfg);
    recs = T.fine.records;
    n = min(5, numel(recs)); lat = zeros(n,1);
    for i = 1:n
        [e, m] = generateLayeredExplanations(recs(i), llm, cfg, feeder);
        lat(i) = m.totalLatencySec;
        T.explanations{i} = e;
        if i == 1
            fprintf('\n  Example 69-bus explanation (record 1):\n  SUMMARY : %s\n  STANDARD: %s\n', e.summary, e.standard);
        end
    end
    T.llmLatency = lat;
    fprintf('  LLM latency, three levels: %.1f +/- %.1f s\n', mean(lat), std(lat));
end

T.finishedAt = datetime('now');
saveable = T; %#ok<NASGU>
save(fullfile(cfg.io.resultsDir, 'transfer69.mat'), '-struct', 'saveable', '-v7.3');
fprintf('\n  Saved %s and %s\n', fullfile(cfg.io.resultsDir,'transfer69.mat'), logFile);
end
