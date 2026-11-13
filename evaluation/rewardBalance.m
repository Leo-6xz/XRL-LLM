function report = rewardBalance(policyFcn, cfg, feeder, opts)
%REWARDBALANCE  Report what fraction of the reward each objective actually drives.
%
%   REPORT = REWARDBALANCE(POLICYFCN, CFG, FEEDER, OPTS)
%
%   WHY THIS EXISTS
%   ---------------
%   A multi-objective reward can be badly miscalibrated without any symptom
%   that looks like a bug. Training converges, the curve is clean, the
%   policy passes DIAGNOSEPOLICY -- and the agent is optimizing one term
%   while ignoring the other three.
%
%   That is what happened on the first successful run here. With
%   alpha=10, beta=0.1, gamma=0.01, delta=0.05 the achieved decomposition was:
%
%       alpha*Vviol   (voltage)          0.9%
%       beta*Ploss    (loss)             2.4%
%       gamma*dAction (wear)             6.0%
%       delta*Interp  (interpretability) 90.8%
%
%   The interpretability penalty was 91% of the signal. The agent
%   responded rationally: it froze the capacitor banks in a fixed
%   configuration and regulated with the tap and inverters only, buying a
%   33% switching reduction at the cost of 52x more voltage violations
%   than the rule baseline.
%
%   THE TRAP: THE BALANCE SHIFTS AS THE POLICY IMPROVES
%   ---------------------------------------------------
%   Weights tuned against the UNCONTROLLED operating point look sensible.
%   But Vviol collapses toward zero as soon as control works, while the
%   interpretability and wear penalties stay roughly constant -- so the
%   term that was dominant at initialisation becomes negligible at
%   convergence, and a term that was negligible takes over. Always
%   re-measure the balance against the TRAINED policy, not the initial one.
%
%   REPORT.MARGINALCOST answers the question that matters operationally:
%   what does a real violation cost relative to the standing penalties? If
%   a ten-bus 1% excursion is cheaper than holding the devices still, the
%   agent will tolerate excursions, and no amount of further training will
%   change that.
%
%   See also COMPUTEREWARD, DIAGNOSEPOLICY, REWARDWEIGHTSWEEP.

if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nEpisodes'), opts.nEpisodes = 20;   end
if ~isfield(opts,'verbose'),   opts.verbose   = true; end
if ~isfield(opts,'seed'),      opts.seed      = cfg.seed + 500; end
if ~isfield(opts,'label'),     opts.label     = 'policy'; end

rs = RandStream('mt19937ar','Seed',opts.seed);

sumV = 0; sumP = 0; sumA = 0; sumI = 0; n = 0; sumR = 0;

for ep = 1:opts.nEpisodes
    [obs, state] = xrlReset(cfg, feeder, rs);
    done = false;
    while ~done
        a = policyFcn(obs(:)');
        [obs, r, done, state, info] = xrlStep(a(:), state, cfg, feeder);
        sumV = sumV + info.Vviol;
        sumP = sumP + info.Ploss;
        sumA = sumA + info.actionChange;
        sumI = sumI + info.interp;
        sumR = sumR + r;
        n = n + 1;
    end
end

if isfield(cfg.reward,'normalize') && cfg.reward.normalize
    R = cfg.reward.ref;  W = cfg.reward.w;
    t.voltage          = W.voltage * (sumV/n) / R.Vviol;
    t.loss             = W.loss    * (sumP/n) / R.Ploss;
    t.wear             = W.wear    * (sumA/n) / R.action;
    t.interpretability = W.interp  * (sumI/n) / R.interp;
    effAlpha = W.voltage / R.Vviol;
else
    t.voltage          = cfg.reward.alpha * sumV / n;
    t.loss             = cfg.reward.beta  * sumP / n;
    t.wear             = cfg.reward.gamma * sumA / n;
    t.interpretability = cfg.reward.delta * sumI / n;
    effAlpha = cfg.reward.alpha;
end

total = t.voltage + t.loss + t.wear + t.interpretability;

report.terms      = t;
report.total      = total;
report.meanReward = sumR / n;
report.share      = structfun(@(v) v/max(total,eps), t, 'UniformOutput', false);
report.rawMeans   = struct('Vviol', sumV/n, 'Ploss', sumP/n, ...
                           'actionChange', sumA/n, 'interp', sumI/n);

% ---------------------------------------------------------------------
% Marginal cost of a real violation versus the standing penalties
% ---------------------------------------------------------------------
standing = t.wear + t.interpretability;
scen = {1, 0.01; 10, 0.01; 10, 0.03; 33, 0.03};
report.marginalCost = struct('nBus',{},'deviation',{},'cost',{},'dominates',{});
for i = 1:size(scen,1)
    c.nBus      = scen{i,1};
    c.deviation = scen{i,2};
    c.cost      = effAlpha * c.nBus * c.deviation^2;
    c.dominates = c.cost > standing;
    report.marginalCost(end+1) = c; %#ok<AGROW>
end
report.standingPenalty = standing;

% A term carrying under 5% of the signal is not being optimised in any
% meaningful sense, whatever the weight nominally says.
report.dominantTerm = '';
[~, k] = max([t.voltage t.loss t.wear t.interpretability]);
names = {'voltage','loss','wear','interpretability'};
report.dominantTerm  = names{k};
report.dominantShare = report.share.(names{k});
report.balanced      = report.dominantShare < 0.60;

% Loss visibility. The second failed run had voltage correctly dominant in
% intent yet still tripled the losses, because the LOSS term was too small
% to notice. This asks directly: what does 100 kW of extra loss cost
% relative to the standing penalties?
extraPU = 0.010;                                   % 100 kW on a 10 MVA base
if isfield(cfg.reward,'normalize') && cfg.reward.normalize
    report.costOf100kW = cfg.reward.w.loss * extraPU / cfg.reward.ref.Ploss;
else
    report.costOf100kW = cfg.reward.beta * extraPU;
end
report.lossVisible = report.costOf100kW > standing;

if opts.verbose
    fprintf('\n--- Reward balance: %s (%d steps) ---\n', opts.label, n);
    fprintf('  %-30s %11s %8s\n','term','per step','share');
    printTerm('alpha*Vviol (voltage)',       t.voltage,          total);
    printTerm('beta*Ploss (loss)',           t.loss,             total);
    printTerm('gamma*dAction (wear)',        t.wear,             total);
    printTerm('delta*Interp (interpretab.)', t.interpretability, total);
    fprintf('  %-30s %11.5f\n', 'total (= -reward)', total);

    fprintf('\n  Marginal cost of a violation vs standing penalty (%.5f):\n', standing);
    for i = 1:numel(report.marginalCost)
        c = report.marginalCost(i);
        if c.dominates, v = 'violation dominates'; else, v = 'PENALTY DOMINATES'; end
        fprintf('    %2d bus(es) %.0f%% outside band -> %.5f   %s\n', ...
            c.nBus, 100*c.deviation, c.cost, v);
    end

    fprintf('\n  Cost of 100 kW extra loss: %.5f  vs standing penalty %.5f  -> %s\n', ...
        report.costOf100kW, standing, ...
        ternaryStr(report.lossVisible, 'loss is visible', ...
                   'LOSS IS INVISIBLE -- the agent can waste energy for free'));

    if ~report.balanced
        fprintf(['\n  WARNING: "%s" carries %.0f%% of the reward. The other\n' ...
                 '  objectives are effectively not being optimised. Rebalance the\n' ...
                 '  weights against THIS policy and retrain -- see the header of\n' ...
                 '  this file for why weights tuned at initialisation mislead.\n'], ...
                 report.dominantTerm, 100*report.dominantShare);
    else
        fprintf('\n  Balance looks reasonable (no term above 60%%).\n');
    end
end
end

function s = ternaryStr(c,a,b)
if c, s = a; else, s = b; end
end

function printTerm(label, v, total)
fprintf('  %-30s %11.5f %7.1f%%\n', label, v, 100*v/max(total,eps));
end
