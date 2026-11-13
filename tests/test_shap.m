function results = test_shap()
%TEST_SHAP  Validate both SHAP engines against known-answer cases.
%
%   The tests here are ordered from strongest to weakest guarantee:
%     1. AXIOMS on an analytic function whose Shapley values are known in
%        closed form (an additive linear model attributes exactly its own
%        coefficients).
%     2. AGREEMENT between the exact engine and KernelSHAP when KernelSHAP
%        enumerates rather than samples -- they must coincide to solver
%        tolerance.
%     3. GROUND TRUTH on a real policy: RULEPOLICY is deliberately LOCAL,
%        so a capacitor channel must be attributed to its own lateral's
%        voltage group. This is the only test that exercises the whole
%        stack against a known answer.

results = testSuite('shap');
cfg = xrlConfig();

% =====================================================================
% 1. Axioms on an additive linear model
% =====================================================================
rng(0);
nFeat = 6; nBG = 40;
w = randn(nFeat,1);
f = @(X) X * w;                       % additive: phi_j = w_j * (x_j - E[bg_j])
bg = randn(nBG, nFeat);
x  = randn(1, nFeat);
groups = num2cell(1:nFeat);

[phiE, outE] = exactSHAP(f, x, bg, groups, struct('maxExactM',16));
expected = (w' .* (x - mean(bg,1)));

results = checkNear(results,'exact: linear model closed form', ...
    max(abs(phiE - expected)), 0, 1e-9);
results = checkNear(results,'exact: efficiency axiom', outE.efficiencyGap, 0, 1e-9);

% Dummy axiom: a feature the model ignores must get exactly zero.
w2 = w; w2(3) = 0;
f2 = @(X) X * w2;
phiD = exactSHAP(f2, x, bg, groups, struct('maxExactM',16));
results = checkNear(results,'exact: dummy feature gets zero', abs(phiD(3)), 0, 1e-10);

% Symmetry axiom: two features entering identically get equal attribution.
w3 = w; w3(1) = 0.7; w3(2) = 0.7;
xs = x; xs(1) = 0.4; xs(2) = 0.4;
bgs = bg; bgs(:,2) = bgs(:,1);
phiS = exactSHAP(@(X) X*w3, xs, bgs, groups, struct('maxExactM',16));
results = checkNear(results,'exact: symmetry axiom', abs(phiS(1)-phiS(2)), 0, 1e-9);

% =====================================================================
% 2. KernelSHAP must reproduce exact values when it enumerates
% =====================================================================
fNL = @(X) tanh(X*w) + 0.5*X(:,1).*X(:,2);     % nonlinear + interaction
phiEx = exactSHAP(fNL, x, bg, groups, struct('maxExactM',16));
[phiK, outK] = kernelSHAP(fNL, x, bg, groups, ...
    struct('nCoalitions', 2^nFeat, 'seed', 1));

results = check(results,'kernelSHAP enumerated for small M', outK.enumerated);
results = checkNear(results,'kernelSHAP == exact (enumerated)', ...
    max(abs(phiK - phiEx)), 0, 1e-6);
results = checkNear(results,'kernelSHAP: efficiency axiom', outK.efficiencyGap, 0, 1e-9);

% Sampled KernelSHAP: looser, but must still satisfy efficiency exactly
% (it is imposed as a constraint, not approximated).
[phiKs, outKs] = kernelSHAP(fNL, x, bg, groups, ...
    struct('nCoalitions', 40, 'seed', 3));
results = checkNear(results,'kernelSHAP sampled: efficiency exact', ...
    outKs.efficiencyGap, 0, 1e-9);
results = check(results,'kernelSHAP sampled: correlates with exact', ...
    corrSafe(phiKs(:), phiEx(:)) > 0.9);

% =====================================================================
% 3. Ground truth on a real, deliberately local policy
% =====================================================================
feeder = buildFeeder(cfg.net.dataFcn());
policyFcn = makePolicyFcn('rule', cfg);

rs = RandStream('mt19937ar','Seed',5);
[obs, ~] = xrlReset(cfg, feeder, rs);

map = featureMapping(cfg);
grpNames = map.groupNames;
grpMem   = map.groupMembers;

% Background from real observations
bgObs = zeros(24, cfg.dims.nObs);
[o, st] = xrlReset(cfg, feeder, rs);
for i = 1:24
    bgObs(i,:) = o(:)';
    [o, ~, d, st] = xrlStep(zeros(9,1), st, cfg, feeder);
    if d, [o, st] = xrlReset(cfg, feeder, rs); end
end

% Channel for the capacitor on bus 30, which sits on Lateral C (26-33).
kCap30 = 1 + find(cfg.cap.buses == 30, 1);
predictFcn = shapValueFunction(policyFcn, kCap30, cfg);
[phiP, outP] = exactSHAP(predictFcn, obs(:)', bgObs, grpMem, ...
    struct('maxExactM', cfg.shap.maxExactM));

results = checkNear(results,'policy SHAP: efficiency axiom', outP.efficiencyGap, 0, 1e-8);

[~, topG] = max(abs(phiP));
isLateralC = contains(grpNames{topG}, 'Lateral C');
results = check(results, ...
    sprintf('bus-30 bank attributed to Lateral C (got "%s")', grpNames{topG}), ...
    isLateralC);

% The same policy's OLTC channel keys on the whole feeder, so it must NOT
% be dominated by a single lateral.
predictOLTC = shapValueFunction(policyFcn, 1, cfg);
phiO = exactSHAP(predictOLTC, obs(:)', bgObs, grpMem, ...
    struct('maxExactM', cfg.shap.maxExactM));
shareTop = max(abs(phiO)) / sum(abs(phiO));
results = check(results,'OLTC attribution is not degenerate', shareTop < 0.95);

% =====================================================================
% 4. Housekeeping
% =====================================================================
top = findTopFeatures(phiP, 3, grpNames);
results = check(results,'findTopFeatures returns 3',      numel(top) == 3);
results = check(results,'findTopFeatures sorted',         top(1).absValue >= top(2).absValue);
results = checkNear(results,'shares sum <= 1', double(sum([top.share]) <= 1+1e-9), 1, 1e-9);
results = check(results,'grouping covers all but V1',     nnz(map.groupIdx == 0) == 1);

% =====================================================================
% 5. Agent policy path -- batched shape handling
% =====================================================================
% The rule policy is a plain MATLAB loop and always returns [n x nAction].
% A Reinforcement Learning Toolbox actor does not: EVALUATE returns
% [outputDim x batchSize x sequenceLength], which collapses to 2-D for a
% single observation and stays 3-D for a batch. Code that transposes it
% blindly passes every single-observation rollout and then fails on the
% first batched SHAP call. This test exercises the batched path directly.
if ~isempty(ver('rl')) && ~isempty(ver('nnet'))
    try
        env  = XRL33Env(cfg, feeder);
        ag   = createPPOAgent(getObservationInfo(env), getActionInfo(env), cfg);
        pf   = makePolicyFcn(ag, cfg);

        a1 = pf(obs(:)');
        results = check(results,'agent policy: single obs -> [1 x nAction]', ...
            isequal(size(a1), [1 cfg.dims.nAction]));

        Xb = repmat(obs(:)', 7, 1);
        Xb(:,1) = linspace(0.96, 1.04, 7);
        ab = pf(Xb);
        results = check(results,'agent policy: batch of 7 -> [7 x nAction]', ...
            isequal(size(ab), [7 cfg.dims.nAction]));
        results = check(results,'agent policy: batch output finite', all(isfinite(ab(:))));

        % Row i of a batch must equal the single-row evaluation of row i,
        % TO SINGLE PRECISION. dlnetwork runs in single by default
        % (eps ~ 1.2e-7) and different batch sizes take different BLAS/GPU
        % kernel paths, so a batched and an unbatched evaluation of the
        % same input agree to roughly eps, not to double precision. An
        % earlier 1e-9 tolerance here was unsatisfiable by construction.
        aSingle = pf(Xb(3,:));
        tolSingle = 10 * eps('single');           % ~1.2e-6
        results = checkNear(results,'agent policy: batch row matches single call', ...
            max(abs(ab(3,:) - aSingle)), 0, tolSingle);

        % Deterministic: the same input twice must give the same action.
        results = checkNear(results,'agent policy: deterministic (policy mean)', ...
            max(abs(pf(Xb(1,:)) - pf(Xb(1,:)))), 0, 1e-12);
    catch ME
        results = check(results, ...
            sprintf('agent policy path (threw: %s)', ME.message), false);
    end
else
    fprintf('  (skipped agent policy checks: RL/DL Toolbox not available)\n');
end

results = report(results);
end

function r = corrSafe(a,b)
a = a - mean(a); b = b - mean(b);
d = sqrt(sum(a.^2)*sum(b.^2));
if d == 0, r = 0; else, r = sum(a.*b)/d; end
end
