function results = rewardWeightSweep(cfg, feeder, opts)
%REWARDWEIGHTSWEEP  Map the performance / interpretability trade-off.
%
%   RESULTS = REWARDWEIGHTSWEEP(CFG, FEEDER, OPTS)
%
%   Sweeps the voltage weight ALPHA and the interpretability weight DELTA,
%   trains a short agent at each point, and records both control
%   performance and explanation concentration.
%
%   THE FIGURE THIS PRODUCES IS OUR PAPER'S CENTRAL CLAIM
%   ----------------------------------------------------
%   The interpretability penalty is only a contribution if it buys
%   explainability at an acceptable cost in control performance. A single
%   operating point (delta = 0.05) cannot show that -- it invites the
%   obvious question "what would have happened at 0.02, or 0.2?". The
%   trade-off CURVE answers it, and it is also the honest way to present a
%   result where the penalty does cost something.
%
%   Plot RESULTS.LOSSKW against RESULTS.TOPSHARE with DELTA as the
%   parameter. The shape to hope for is a knee: concentration rising
%   steeply while loss barely moves, then flattening. If instead it is a
%   straight line, the penalty is simply trading one objective for another
%   and there is no free lunch to claim.
%
%   COST WARNING
%   ------------
%   This trains one agent per grid point. With the defaults below that is
%   9 agents. Keep OPTS.EPISODES low (a few hundred) for the sweep and
%   retrain the chosen point at full length -- the sweep needs to rank
%   settings, not to produce the final policy.
%
%   See also TRAINPPO, CONTROLMETRICS, COLLECTATTRIBUTIONS.

if nargin < 1 || isempty(cfg),    cfg = xrlConfig(); end
if nargin < 2 || isempty(feeder), feeder = buildFeeder(cfg.net.dataFcn()); end
if nargin < 3 || isempty(opts),   opts = struct(); end
% Default ranges depend on the reward form: the scale-free weights are
% relative priorities (voltage = 1.0, interp = 0.10 by default), whereas
% the legacy raw weights are unit-bearing gains (alpha = 10, delta = 0.05).
% Passing raw-weight ranges to a normalised run silently produces a
% meaningless sweep.
useNorm = isfield(cfg.reward,'normalize') && cfg.reward.normalize;
if ~isfield(opts,'alphaRange')
    if useNorm, opts.alphaRange = 1.0;           else, opts.alphaRange = [5 10 20];     end
end
if ~isfield(opts,'deltaRange')
    if useNorm, opts.deltaRange = [0 0.05 0.10 0.20 0.40];
    else,       opts.deltaRange = [0 0.05 0.20]; end
end
if ~isfield(opts,'episodes'),   opts.episodes   = 400;            end
if ~isfield(opts,'evalEps'),    opts.evalEps    = 20;             end
if ~isfield(opts,'shapEps'),    opts.shapEps    = 3;              end
if ~isfield(opts,'verbose'),    opts.verbose    = true;           end

nA = numel(opts.alphaRange);
nD = numel(opts.deltaRange);

results.alpha    = opts.alphaRange;
results.delta    = opts.deltaRange;
results.violRate = nan(nA,nD);
results.lossKW   = nan(nA,nD);
results.switching= nan(nA,nD);
results.interp   = nan(nA,nD);
results.topShare = nan(nA,nD);
results.reward   = nan(nA,nD);

for ia = 1:nA
    for id = 1:nD
        a = opts.alphaRange(ia);
        d = opts.deltaRange(id);

        if opts.verbose
            fprintf('\n=== Sweep point alpha=%.3g delta=%.3g (%d/%d) ===\n', ...
                a, d, (ia-1)*nD+id, nA*nD);
        end

        c = cfg;
        % Vary the weight that the ACTIVE reward form actually reads.
        %
        % An earlier version always set the legacy raw weights
        % (c.reward.alpha / c.reward.delta). Under the scale-free form
        % (cfg.reward.normalize = true) COMPUTEREWARD reads cfg.reward.w
        % and ignores those fields entirely, so every delta > 0 grid point
        % trained an IDENTICAL agent -- four points agreeing to 52.827 kW,
        % which is the signature of a parameter that is not being applied.
        % Only delta = 0 differed, and only because a separate branch
        % switched the penalty off.
        if isfield(c.reward,'normalize') && c.reward.normalize
            c.reward.w.voltage = a;      % priority weights, not raw gains
            c.reward.w.interp  = d;
        else
            c.reward.alpha     = a;
            c.reward.delta     = d;
        end
        c.train.maxEpisodes  = opts.episodes;
        c.train.plots        = 'none';
        c.io.verbose         = false;
        c.train.savePath     = fullfile(cfg.io.resultsDir, ...
            sprintf('sweep_a%g_d%g.mat', a, d));

        try
            agent = trainPPO(c, feeder);
            pf = makePolicyFcn(agent, c);
        catch ME
            warning('rewardWeightSweep:trainFailed', ...
                'Training failed at alpha=%g delta=%g (%s); skipping.', a, d, ME.message);
            continue
        end

        m = controlMetrics(pf, c, feeder, opts.evalEps, ...
            struct('verbose',false,'seed',cfg.seed+1000, ...
                   'label',sprintf('a%g d%g',a,d)));

        results.violRate(ia,id)  = m.violationRate;
        results.lossKW(ia,id)    = m.meanLossKW;
        results.switching(ia,id) = m.switchingPerStep;
        results.interp(ia,id)    = m.interpPenalty;
        results.reward(ia,id)    = m.meanReward;

        % Explanation concentration: the mean share carried by the single
        % top-ranked factor. This is the quantity the penalty is supposed
        % to raise, and it is measurable without any LLM in the loop.
        try
            recs = collectAttributions(pf, c, feeder, opts.shapEps, ...
                struct('maxRecords',20,'verbose',false));
            if ~isempty(recs)
                results.topShare(ia,id) = mean(arrayfun(@(r) r.topFeatures(1).share, recs));
            end
        catch ME
            warning('rewardWeightSweep:shapFailed', ...
                'SHAP failed at alpha=%g delta=%g (%s).', a, d, ME.message);
        end

        if opts.verbose
            fprintf('  viol %.3f%%  loss %.1f kW  switching %.3f  topShare %.3f\n', ...
                100*results.violRate(ia,id), results.lossKW(ia,id), ...
                results.switching(ia,id), results.topShare(ia,id));
        end
    end
end

% Degeneracy check. If the swept parameter is not reaching the reward,
% every grid point trains the same agent and the "curve" is a flat line
% that looks like a legitimate null result.
uniqLoss = unique(round(results.lossKW(~isnan(results.lossKW)), 3));
if numel(uniqLoss) < max(2, 0.5*numel(opts.deltaRange))
    warning('rewardWeightSweep:degenerate', ...
        ['Only %d distinct loss value(s) across %d grid points. The swept ', ...
         'weight is probably not reaching COMPUTEREWARD -- check that the ', ...
         'field being varied matches cfg.reward.normalize (%d). Do NOT read ', ...
         'this as a flat trade-off curve.'], ...
        numel(uniqLoss), numel(results.lossKW), useNorm);
end

if opts.verbose
    fprintf('\n--- Sweep summary (rows = %s, cols = %s) ---\n', ...
        tern2(useNorm,'w.voltage','alpha'), tern2(useNorm,'w.interp','delta'));
    printGrid('Violation rate (%%)', 100*results.violRate, results.alpha, results.delta);
    printGrid('Mean loss (kW)',      results.lossKW,       results.alpha, results.delta);
    printGrid('Switching per step',  results.switching,    results.alpha, results.delta);
    printGrid('Top-1 SHAP share',    results.topShare,     results.alpha, results.delta);
end
end

function printGrid(name, M, rows, cols)
fprintf('\n  %s\n', name);
fprintf('        ');
fprintf('%10.3g', cols); fprintf('   <- delta\n');
for i = 1:size(M,1)
    fprintf('  %5.3g ', rows(i));
    fprintf('%10.3f', M(i,:));
    fprintf('\n');
end
end

function s = tern2(c,a,b)
if c, s = a; else, s = b; end
end
