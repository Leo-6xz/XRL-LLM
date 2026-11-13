function results = controlMetrics(policyFcn, cfg, feeder, nEpisodes, opts)
%CONTROLMETRICS  Closed-loop control performance of a policy.
%
%   RESULTS = CONTROLMETRICS(POLICYFCN, CFG, FEEDER, NEPISODES, OPTS)
%
%   Rolls the policy out over NEPISODES fresh 24-hour scenarios and reports
%   the control-side numbers, alongside the SAME scenarios evaluated with
%   no control at all (all devices neutral).
%
%   RESULTS fields
%   --------------
%     .violationRate      fraction of bus-steps outside the ANSI band
%     .violationRateUnc   ditto, uncontrolled
%     .violationReduction 1 - controlled/uncontrolled
%     .overvoltageShare   fraction of violations that are HIGH rather than low
%     .meanLossKW         average active loss per step
%     .meanLossKWUnc      ditto, uncontrolled
%     .lossReduction
%     .switchingPerStep   mean L1 action change (device wear proxy)
%     .interpPenalty      mean interpretability penalty
%     .meanReward
%     .byScenario         breakdown per operating regime
%
%   Run it both ways:
%       rl   = controlMetrics(makePolicyFcn(agent,cfg), cfg, feeder, 50);
%       rule = controlMetrics(makePolicyFcn('rule',cfg), cfg, feeder, 50);
%
%   Use the same OPTS.SEED for both so the scenarios are identical --
%   otherwise the comparison is confounded by scenario luck.
%
%   See also RULEPOLICY, MAKEPOLICYFCN.

if nargin < 4 || isempty(nEpisodes), nEpisodes = 50; end
if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'seed'),    opts.seed    = cfg.seed + 1000; end
if ~isfield(opts,'verbose'), opts.verbose = cfg.io.verbose;  end
if ~isfield(opts,'label'),   opts.label   = 'policy';        end

rs = RandStream('mt19937ar','Seed',opts.seed);

nBus = feeder.nBus;
nViolC = 0; nViolU = 0; nOver = 0; nBusSteps = 0;
lossC = 0; lossU = 0; nSteps = 0;
switching = 0; interp = 0; totalReward = 0;

scenTags = {}; scenViolC = []; scenViolU = []; scenLoss = [];

neutral = zeros(cfg.dims.nAction, 1);

for ep = 1:nEpisodes
    [obs, state] = xrlReset(cfg, feeder, rs);
    done = false;

    epViolC = 0; epViolU = 0; epLoss = 0; epSteps = 0;
    tag = state.prof.scenarioTag;

    while ~done
        preState = state;

        % Uncontrolled counterfactual on the SAME load/PV realisation.
        devU = decodeAction(neutral, cfg, preState.pvP);
        pfU  = solveWithDevices(preState, devU, cfg, feeder);
        vU   = pfU.Vm;

        a = policyFcn(obs(:)');
        [obs, reward, done, state, info] = xrlStep(a(:), state, cfg, feeder);

        vC = info.VmPost;

        nViolC = nViolC + nnz(vC < cfg.limits.Vmin | vC > cfg.limits.Vmax);
        nOver  = nOver  + nnz(vC > cfg.limits.Vmax);
        nViolU = nViolU + nnz(vU < cfg.limits.Vmin | vU > cfg.limits.Vmax);
        nBusSteps = nBusSteps + nBus;

        lossC = lossC + info.PlossKW;
        lossU = lossU + pfU.Ploss * cfg.net.baseMVA * 1000;

        switching   = switching + info.actionChange;
        interp      = interp + info.interp;
        totalReward = totalReward + reward;
        nSteps      = nSteps + 1;

        epViolC = epViolC + nnz(vC < cfg.limits.Vmin | vC > cfg.limits.Vmax);
        epViolU = epViolU + nnz(vU < cfg.limits.Vmin | vU > cfg.limits.Vmax);
        epLoss  = epLoss + info.PlossKW;
        epSteps = epSteps + 1;
    end

    scenTags{end+1}  = tag;               %#ok<AGROW>
    scenViolC(end+1) = epViolC / (epSteps*nBus); %#ok<AGROW>
    scenViolU(end+1) = epViolU / (epSteps*nBus); %#ok<AGROW>
    scenLoss(end+1)  = epLoss / epSteps;  %#ok<AGROW>
end

results.label             = opts.label;
results.nEpisodes         = nEpisodes;
results.nSteps            = nSteps;

results.violationRate     = nViolC / nBusSteps;
results.violationRateUnc  = nViolU / nBusSteps;
results.violationReduction= 1 - results.violationRate / max(results.violationRateUnc, eps);
results.overvoltageShare  = nOver / max(nViolC, 1);

results.meanLossKW        = lossC / nSteps;
results.meanLossKWUnc     = lossU / nSteps;
results.lossReduction     = 1 - results.meanLossKW / max(results.meanLossKWUnc, eps);

results.switchingPerStep  = switching / nSteps;
results.interpPenalty     = interp / nSteps;
results.meanReward        = totalReward / nSteps;

results.passesViolTarget  = results.violationReduction >= cfg.eval.targetViolRedux;

% Per-scenario breakdown
tags = unique(scenTags);
results.byScenario = struct('tag',{},'n',{},'violationRate',{}, ...
                            'violationRateUnc',{},'meanLossKW',{});
for i = 1:numel(tags)
    m = strcmp(scenTags, tags{i});
    s.tag              = tags{i};
    s.n                = nnz(m);
    s.violationRate    = mean(scenViolC(m));
    s.violationRateUnc = mean(scenViolU(m));
    s.meanLossKW       = mean(scenLoss(m));
    results.byScenario(end+1) = s;
end

if opts.verbose
    fprintf('\n--- Control performance: %s (%d episodes) ---\n', opts.label, nEpisodes);
    fprintf('  Violated bus-steps : %.3f%%  (uncontrolled %.3f%%)\n', ...
        100*results.violationRate, 100*results.violationRateUnc);
    fprintf('  Violation reduction: %.1f%%  (target >= %.0f%%) %s\n', ...
        100*results.violationReduction, 100*cfg.eval.targetViolRedux, ...
        ternary(results.passesViolTarget,'[PASS]','[BELOW TARGET]'));
    fprintf('    of which overvoltage: %.0f%%\n', 100*results.overvoltageShare);
    fprintf('  Mean loss          : %.1f kW  (uncontrolled %.1f kW, -%.1f%%)\n', ...
        results.meanLossKW, results.meanLossKWUnc, 100*results.lossReduction);
    fprintf('  Switching per step : %.3f\n', results.switchingPerStep);
    fprintf('  Interp. penalty    : %.3f\n', results.interpPenalty);
    fprintf('  Mean reward        : %.3f\n', results.meanReward);
    for i = 1:numel(results.byScenario)
        b = results.byScenario(i);
        fprintf('    %-14s n=%3d  violations %.3f%% (unc %.3f%%)  loss %.1f kW\n', ...
            b.tag, b.n, 100*b.violationRate, 100*b.violationRateUnc, b.meanLossKW);
    end
end
end

function out = ternary(c,a,b)
if c, out = a; else, out = b; end
end
