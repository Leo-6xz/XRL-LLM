function report = diagnosePolicy(policyFcn, cfg, feeder, opts)
%DIAGNOSEPOLICY  Detect a collapsed or saturated policy BEFORE trusting it.
%
%   REPORT = DIAGNOSEPOLICY(POLICYFCN, CFG, FEEDER, OPTS)
%
%   WHY THIS EXISTS
%   ---------------
%   A PPO run can complete all 6000 episodes, print a plausible-looking
%   reward curve, and produce a policy that is entirely broken. That
%   happened here: a misplaced optimiser setting left the actor training at
%   100x the intended learn rate, the tanh mean path saturated, and the
%   agent froze holding every device at maximum injection.
%
%   The reward curve did NOT make this obvious -- it looked flat, which is
%   ambiguous between "converged" and "dead". The statistic that made it
%   unambiguous was SWITCHING PER STEP = 0.000: an agent that never changes
%   its action is not a controller. This function checks that and four
%   other cheap signals, on a handful of rollouts, in about a second.
%
%   CHECKS
%   ------
%     actionRange     spread of each action channel across states. Near
%                     zero on all channels means the policy ignores its
%                     input entirely.
%     saturatedFrac   fraction of action components pinned at |a| > 0.99.
%                     High values mean the tanh mean path has saturated and
%                     gradients through it have vanished.
%     switchingRate   mean L1 change between consecutive actions. Exactly
%                     zero is the collapse signature.
%     stateSensitivity change in action for a large change in input. Zero
%                     means the network is not reading the observation.
%     violationRate   quick control sanity check.
%
%   Call it on any policy, trained or rule-based, before spending time on
%   SHAP or explanations.
%
%   See also TRAINPPO, MAKEPOLICYFCN, CONTROLMETRICS.

if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nEpisodes'), opts.nEpisodes = 5;    end
if ~isfield(opts,'verbose'),   opts.verbose   = true; end
if ~isfield(opts,'seed'),      opts.seed      = cfg.seed + 99; end

rs = RandStream('mt19937ar','Seed',opts.seed);

A = []; obsAll = []; switching = 0; nStep = 0; nViol = 0; nBusStep = 0;

for ep = 1:opts.nEpisodes
    [obs, state] = xrlReset(cfg, feeder, rs);
    done = false; aPrev = [];
    while ~done
        a = policyFcn(obs(:)');
        a = a(:)';
        A(end+1,:)      = a;      %#ok<AGROW>
        obsAll(end+1,:) = obs(:)'; %#ok<AGROW>
        if ~isempty(aPrev)
            switching = switching + sum(abs(a - aPrev));
        end
        aPrev = a;
        [obs, ~, done, state, info] = xrlStep(a(:), state, cfg, feeder);
        nViol    = nViol + nnz(info.VmPost < cfg.limits.Vmin | info.VmPost > cfg.limits.Vmax);
        nBusStep = nBusStep + feeder.nBus;
        nStep    = nStep + 1;
    end
end

report.nSteps         = nStep;
report.actionRange    = max(A,[],1) - min(A,[],1);
report.actionMean     = mean(A,1);
report.saturatedFrac  = mean(abs(A(:)) > 0.99);
report.switchingRate  = switching / max(nStep-opts.nEpisodes, 1);
report.violationRate  = nViol / nBusStep;

% State sensitivity: drive the voltage channels hard and see if anything
% moves. A healthy Volt/VAr policy MUST respond to this.
oLow  = obsAll(1,:); oLow(1:cfg.dims.nBus)  = 0.94;
oHigh = obsAll(1,:); oHigh(1:cfg.dims.nBus) = 1.06;
report.stateSensitivity = max(abs(policyFcn(oHigh) - policyFcn(oLow)));

% ---------------------------------------------------------------------
% Verdicts
% ---------------------------------------------------------------------
report.frozen      = report.switchingRate < 1e-9;
report.saturated   = report.saturatedFrac > 0.80;
report.unresponsive= report.stateSensitivity < 1e-3;
report.deadChannels= find(report.actionRange < 1e-6);
report.healthy     = ~report.frozen && ~report.saturated && ~report.unresponsive;

if opts.verbose
    fprintf('\n--- Policy diagnostic (%d steps) ---\n', nStep);
    fprintf('  Switching per step   : %.4f  %s\n', report.switchingRate, ...
        flag(report.frozen, 'FROZEN -- the policy never changes its action'));
    fprintf('  Saturated components : %.1f%%  %s\n', 100*report.saturatedFrac, ...
        flag(report.saturated, 'SATURATED -- tanh is pinned, gradients are dead'));
    fprintf('  State sensitivity    : %.4f  %s\n', report.stateSensitivity, ...
        flag(report.unresponsive, 'UNRESPONSIVE -- ignores the observation'));
    fprintf('  Violation rate       : %.2f%%\n', 100*report.violationRate);
    if ~isempty(report.deadChannels)
        fprintf('  Dead action channels : %s\n', mat2str(report.deadChannels));
    end
    fprintf('  Mean action per channel: %s\n', mat2str(round(report.actionMean,2)));
    if report.healthy
        fprintf('  VERDICT: policy looks healthy.\n');
    else
        fprintf(['  VERDICT: POLICY IS BROKEN. Do not run the explanation stages.\n' ...
                 '  Most likely causes, in order:\n' ...
                 '    1. Actor learn rate too high (check the printout from\n' ...
                 '       createPPOAgent -- it should read 1.0e-04, not 1.0e-02).\n' ...
                 '    2. Entropy weight too low; raise cfg.ppo.entropyWeight.\n' ...
                 '    3. Reward scale too large, saturating the advantage.\n']);
    end
end
end

function s = flag(bad, msg)
if bad, s = ['<-- ' msg]; else, s = ''; end
end
