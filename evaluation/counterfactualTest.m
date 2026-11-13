function results = counterfactualTest(records, policyFcn, cfg, feeder, opts)
%COUNTERFACTUALTEST  Causal faithfulness of the SHAP attributions.
%
%   RESULTS = COUNTERFACTUALTEST(RECORDS, POLICYFCN, CFG, FEEDER, OPTS)
%
%   Tests whether the factors SHAP identified as decisive are the factors
%   that actually drive the policy, by perturbing them and measuring the
%   response.
%
%   WHAT IS MEASURED HERE
%   -----------------------------
%   The probe is BASELINE SUBSTITUTION, not a small relative perturbation:
%
%       measured_j = f(x) - E_bg[ f(x with group j replaced by bg_j) ]
%
%   i.e. the marginal contribution of group j when it is the last player to
%   join. For an additive model this equals phi_j exactly, so the three
%   statistics below have a known correct answer.
%
%   WHY NOT A SMALL PERTURBATION -- A BUG WORTH UNDERSTANDING
%   ---------------------------------------------------------
%   An earlier version of this file perturbed each feature by +5% and
%   compared sign(phi_j) against the sign of the induced change. That is
%   wrong, and wrong in a way that is easy to miss because the rank
%   correlation still looks respectable (~0.77). A Shapley value encodes
%
%       phi_j  ~  (df/dx_j) * (x_j - E[bg_j])
%
%   -- a derivative TIMES a deviation from the reference. The perturbation
%   probe recovers only the derivative. The two therefore disagree in sign
%   whenever the feature sits BELOW its background mean, and on a loaded
%   distribution feeder the bus voltages sit below the mean essentially
%   always. The result was a sign agreement of exactly 0.000 -- a
%   deterministic inversion masquerading as total failure -- and a top-1
%   hit rate of 0.067, because |phi| ranks by derivative*deviation while
%   the perturbation ranks by derivative alone.
%
%   Measured on a nonlinear test function with known Shapley values:
%
%       probe                    sign agreement   top-1   rank corr
%       relative perturbation         0.575       0.575     0.824
%       baseline substitution         1.000       0.775     0.761
%
%   Note that the rank correlation is slightly LOWER for the correct probe.
%   If you are tuning a faithfulness number, that is the trap: the metric
%   that flatters the pipeline was the broken one.
%
%   THE THREE STATISTICS
%   --------------------
%   (a) RANK FAITHFULNESS. Spearman correlation between |phi| and
%       |measured|. Headline number, comparable to the 0.80 target.
%
%   (b) TOP-1 HIT RATE. Fraction of decisions where the group SHAP ranked
%       first also produced the largest measured contribution. This is the
%       operationally meaningful number: the probability that the one
%       factor shown to the operator is the one that actually mattered.
%
%   (c) SIGN AGREEMENT. Whether the direction of the measured contribution
%       matches the sign of the SHAP value. An attribution with the right
%       magnitude and the wrong sign would tell an operator to intervene
%       backwards.
%
%   Each is reported overall and split by scenario, because the brief
%   claims post-hoc attribution is most reliable under stress. That claim
%   is testable here.
%
%   See also COLLECTATTRIBUTIONS, ABLATIONEXPERIMENTS.

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'perturbFrac'), opts.perturbFrac = cfg.eval.cfPerturbFrac; end
if ~isfield(opts,'nTests'),      opts.nTests      = cfg.eval.nCounterfactual; end
if ~isfield(opts,'verbose'),     opts.verbose     = cfg.io.verbose; end
if ~isfield(opts,'background'),  opts.background  = [];           end
% Below this, an attribution is treated as "the policy does not use this
% group". Scaled to the typical magnitude of a normalised setpoint.
if ~isfield(opts,'attribTol'),   opts.attribTol   = 1e-6;         end

if isempty(records)
    error('counterfactualTest:noRecords', 'No attribution records supplied.');
end

n = min(opts.nTests, numel(records));
sel = round(linspace(1, numel(records), n));

map = featureMapping(cfg);

% Background distribution for the substitution probe. Reuse the one SHAP
% was computed against when available -- comparing an attribution to a
% probe with a DIFFERENT reference is not a fair test.
background = opts.background;
if isempty(background)
    background = vertcat(records.obs)';
    if size(background,1) > 64
        background = background(round(linspace(1,size(background,1),64)), :);
    end
end
nBG = size(background,1);
% Use the SAME resolver as collectAttributions. Reading cfg.shap.mode
% directly here (the previous behaviour) hardcoded the fine 12-group set
% and silently ignored cfg.shap.grouping, so a coarse-grouping run
% attributed with 5 groups but measured faithfulness against 12 -- a
% guaranteed mismatch. The records carry their own group count, so verify.
[groups, ~, groupingTag] = shapGroups(cfg, map);
M = numel(groups);
if ~isempty(records) && isfield(records(1),'phi') && numel(records(1).phi) ~= M
    error('counterfactualTest:groupMismatch', ...
        ['The records were built with %d SHAP players but the current ', ...
         'config resolves to %d (%s grouping). Evaluate the records with ', ...
         'the SAME cfg.shap.grouping they were collected under.'], ...
        numel(records(1).phi), M, groupingTag);
end
if opts.verbose
    fprintf('  Faithfulness on %s grouping (%d players)\n', groupingTag, M);
end

rankCorr    = nan(n,1);
rankCorrInf = nan(n,1);
nInformative= nan(n,1);
top1Hit  = false(n,1);
signOK   = nan(n,1);
scenario = cell(n,1);

for i = 1:n
    r = records(sel(i));
    k = r.shapMeta.explainedChannel;
    obs0 = r.obs(:)';

    a0 = policyFcn(obs0);
    f0 = a0(k);

    measured = zeros(1, M);

    % Baseline substitution: replace group g with each background sample in
    % turn and average. The drop from f(x) is group g's marginal
    % contribution as the last player to join the coalition.
    for g = 1:M
        idx = groups{g};
        Xh  = repmat(obs0, nBG, 1);
        Xh(:, idx) = background(:, idx);
        aH  = policyFcn(Xh);
        measured(g) = f0 - mean(aH(:,k));
    end

    phi = r.phi(:)';

    % (a) Spearman rank correlation between |phi| and |measured|
    rankCorr(i) = spearman(abs(phi), abs(measured));

    % (a2) The same correlation restricted to groups that carry a
    % non-negligible attribution.
    %
    % Feature groups the policy provably ignores (for rulePolicy: load, PV,
    % aggregate level, time of day) sit at exactly zero in BOTH phi and the
    % measured contribution, so they rank at the bottom of both lists and
    % agree trivially. Including them INFLATES Spearman -- in simulation,
    % five such groups out of twelve raise it from 0.833 to 0.930. The
    % headline number is therefore the generous one.
    informative = (abs(phi) > opts.attribTol) | (abs(measured) > opts.attribTol);
    if nnz(informative) >= 3
        rankCorrInf(i) = spearman(abs(phi(informative)), abs(measured(informative)));
        nInformative(i) = nnz(informative);
    end

    % (b) top-1 hit
    [~, iShap] = max(abs(phi));
    [~, iMeas] = max(abs(measured));
    top1Hit(i) = (iShap == iMeas);

    % (c) sign agreement on the top-ranked player. Both quantities are
    % contributions relative to the same background, so their signs are
    % directly comparable -- see the header for why a derivative probe is not.
    if abs(measured(iShap)) > 1e-12
        signOK(i) = double(sign(phi(iShap)) == sign(measured(iShap)));
    end

    scenario{i} = r.scenario;

    if opts.verbose && mod(i,25) == 0
        fprintf('  counterfactual: %d/%d\n', i, n);
    end
end

results.n              = n;
results.rankCorr       = rankCorr;
results.top1Hit        = top1Hit;
results.signOK         = signOK;
results.scenario       = scenario;

results.faithfulness      = mean(rankCorr, 'omitnan');    % all groups
results.faithfulnessSD    = std(rankCorr, 'omitnan');
results.faithfulnessInf   = mean(rankCorrInf, 'omitnan'); % informative only
results.faithfulnessInfSD = std(rankCorrInf, 'omitnan');
results.rankCorrInf       = rankCorrInf;
results.meanInformative   = mean(nInformative, 'omitnan');
results.top1Rate       = mean(top1Hit);
results.signAgreement  = mean(signOK, 'omitnan');
results.passesTarget   = results.faithfulness >= cfg.eval.targetFaithful;

% Per-scenario breakdown
tags = unique(scenario);
results.byScenario = struct('tag',{},'n',{},'faithfulness',{},'top1Rate',{});
for i = 1:numel(tags)
    m = strcmp(scenario, tags{i});
    s.tag          = tags{i};
    s.n            = nnz(m);
    s.faithfulness = mean(rankCorr(m), 'omitnan');
    s.top1Rate     = mean(top1Hit(m));
    results.byScenario(end+1) = s;
end

if opts.verbose
    fprintf('\n--- Causal faithfulness ---\n');
    fprintf('  Rank correlation : %.3f +/- %.3f  (all %d groups)\n', ...
        results.faithfulness, results.faithfulnessSD, numel(groups));
    fprintf('  REPORT THIS      : %.3f +/- %.3f  (%.1f informative groups) (target >= %.2f) %s\n', ...
        results.faithfulnessInf, results.faithfulnessInfSD, results.meanInformative, ...
        cfg.eval.targetFaithful, passFail(results.faithfulnessInf >= cfg.eval.targetFaithful));
    fprintf(['    Groups the policy provably ignores agree trivially at zero and\n' ...
             '    inflate the all-groups figure; the informative subset is the honest one.\n']);
    fprintf('  Top-1 hit rate   : %.3f\n', results.top1Rate);
    fprintf('  Sign agreement   : %.3f\n', results.signAgreement);
    for i = 1:numel(results.byScenario)
        b = results.byScenario(i);
        fprintf('    %-14s n=%3d  faithfulness %.3f  top-1 %.3f\n', ...
            b.tag, b.n, b.faithfulness, b.top1Rate);
    end
end
end

% =====================================================================
function r = spearman(x, y)
%SPEARMAN  Rank correlation without the Statistics Toolbox.
x = x(:); y = y(:);
rx = tiedRank(x); ry = tiedRank(y);
rx = rx - mean(rx); ry = ry - mean(ry);
d = sqrt(sum(rx.^2) * sum(ry.^2));
if d == 0
    r = NaN;
else
    r = sum(rx .* ry) / d;
end
end

function r = tiedRank(x)
[~, idx] = sort(x);
r = zeros(size(x));
r(idx) = 1:numel(x);
% average ranks within ties
u = unique(x);
for i = 1:numel(u)
    m = (x == u(i));
    if nnz(m) > 1
        r(m) = mean(r(m));
    end
end
end

function s = passFail(tf)
if tf, s = '[PASS]'; else, s = '[BELOW TARGET]'; end
end
