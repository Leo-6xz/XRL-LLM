function results = test_reward()
%TEST_REWARD  Validate the reward and the interpretability penalty.
%
%   The penalty is our paper's first claimed innovation, so its behavior
%   needs to be pinned down by tests rather than asserted in prose. Each
%   test below corresponds to one clause of the definition.

results = testSuite('reward');
cfg    = xrlConfig();
feeder = buildFeeder(cfg.net.dataFcn());
rs     = RandStream('mt19937ar','Seed',3);
[~, state] = xrlReset(cfg, feeder, rs);

% --- Reward monotonicity ----------------------------------------------
dev = decodeAction(zeros(9,1), cfg, state.pvP);
pfGood = struct('Vm', ones(33,1), 'Ploss', 0.01, 'converged', true);
pfBad  = struct('Vm', [ones(32,1); 0.85], 'Ploss', 0.01, 'converged', true);
rGood = computeReward(state, pfGood, dev, [], feeder, cfg);
rBad  = computeReward(state, pfBad,  dev, [], feeder, cfg);
results = check(results,'violation lowers reward', rBad < rGood);

pfLossy = pfGood; pfLossy.Ploss = 0.05;
results = check(results,'higher loss lowers reward', ...
    computeReward(state, pfLossy, dev, [], feeder, cfg) < rGood);

pfNC = pfGood; pfNC.converged = false;
results = checkNear(results,'non-convergence returns -100', ...
    computeReward(state, pfNC, dev, [], feeder, cfg), -100, 1e-9);

% Squared violation: one deep excursion must cost more than several shallow
% ones of the same total magnitude.
pfDeep = pfGood;    pfDeep.Vm(5)  = 0.95 - 0.03;
pfShallow = pfGood; pfShallow.Vm(5:7) = 0.95 - 0.01;
results = check(results,'deep violation costs more than 3 shallow', ...
    computeReward(state, pfDeep, dev, [], feeder, cfg) < ...
    computeReward(state, pfShallow, dev, [], feeder, cfg));

% --- Interpretability penalty -----------------------------------------
stateLow = state; stateLow.Vm = 0.93 * ones(33,1);

a1 = zeros(9,1); a1(1+find(cfg.cap.buses==30,1)) = 1;      % one device
aMany = ones(9,1);                                          % everything

d1 = decodeAction(a1,    cfg, state.pvP);
dM = decodeAction(aMany, cfg, state.pvP);
d0 = decodeAction(zeros(9,1), cfg, state.pvP);

p1 = computeInterpretabilityPenalty(stateLow, d1, d0, feeder, cfg);
pM = computeInterpretabilityPenalty(stateLow, dM, d0, feeder, cfg);
results = check(results,'moving many devices costs more than one', pM > p1);

pNone = computeInterpretabilityPenalty(stateLow, d0, d0, feeder, cfg);
results = checkNear(results,'no movement -> zero penalty', pNone, 0, 1e-12);

% Physical consistency: injecting VArs where voltage is already HIGH must
% cost more than injecting where it is LOW.
stateHigh = state; stateHigh.Vm = 1.04 * ones(33,1);
pLow  = computeInterpretabilityPenalty(stateLow,  d1, d0, feeder, cfg);
pHigh = computeInterpretabilityPenalty(stateHigh, d1, d0, feeder, cfg);
results = check(results,'VAr injection into high voltage costs more', pHigh > pLow);

% Spatial coherence: two distant devices cost more than two adjacent ones,
% when only one end needs help.
stateMix = state; stateMix.Vm = 1.00*ones(33,1); stateMix.Vm(26:33) = 0.94;
aFar = zeros(9,1);
aFar(1+find(cfg.cap.buses==30,1)) = 1;    % Lateral C, needs help
aFar(1+find(cfg.cap.buses==14,1)) = 1;    % main feeder, does not
aNear = zeros(9,1);
aNear(1+find(cfg.cap.buses==30,1)) = 1;
aNear(1+find(cfg.cap.buses==32,1)) = 1;   % same lateral
[pFar,  bFar]  = computeInterpretabilityPenalty(stateMix, decodeAction(aFar, cfg,state.pvP), d0, feeder, cfg);
[pNear, bNear] = computeInterpretabilityPenalty(stateMix, decodeAction(aNear,cfg,state.pvP), d0, feeder, cfg);
results = check(results,'distant co-movement costs more than adjacent', pFar > pNear);
results = check(results,'both report 2 devices moved', ...
    bFar.nMoved == 2 && bNear.nMoved == 2);

% A zero weight must remove the term from the REWARD but must NOT destroy
% the diagnostic breakdown -- COLLECTATTRIBUTIONS selects steps to explain
% via interpBreakdown.nMoved, so stubbing it out silently disabled SHAP
% collection whenever the weight was zero.
cfg0 = cfg; cfg0.reward.normalize = false; cfg0.reward.delta = 0;
[~, info0] = computeReward(stateLow, pfGood, dM, d0, feeder, cfg0);
results = checkNear(results,'zero weight removes the term from the reward', ...
    info0.termInterp, 0, 1e-12);
results = check(results,'zero weight still reports nMoved (diagnostic intact)', ...
    info0.interpBreakdown.nMoved > 0);
results = check(results,'zero weight still reports a raw penalty value', ...
    info0.interp > 0);

% --- Decomposition adds up --------------------------------------------
% Reconstruct from the WEIGHTED term fields, not from the raw weights.
% The earlier version of this test rebuilt the sum using alpha/beta/
% gamma/delta directly, so it silently became wrong the moment the
% scale-free form (cfg.reward.normalize) became the default -- it was then
% comparing the new reward against the old formula. Using info.term*
% keeps the test valid under either formulation.
[rr, ii] = computeReward(stateLow, pfBad, dM, d0, feeder, cfg);
recon = -cfg.reward.scale * ...
    (ii.termVoltage + ii.termLoss + ii.termWear + ii.termInterp);
results = checkNear(results,'reward equals its decomposition (default form)', ...
    rr, recon, 1e-10);

% Both formulations must be internally consistent, and they must disagree
% with each other -- if they agreed, normalization would be a no-op.
cfgN = cfg; cfgN.reward.normalize = true;
cfgR = cfg; cfgR.reward.normalize = false;

[rN, iN] = computeReward(stateLow, pfBad, dM, d0, feeder, cfgN);
[rR, iR] = computeReward(stateLow, pfBad, dM, d0, feeder, cfgR);

results = checkNear(results,'normalised form self-consistent', rN, ...
    -cfgN.reward.scale*(iN.termVoltage+iN.termLoss+iN.termWear+iN.termInterp), 1e-10);
results = checkNear(results,'raw-weight form self-consistent', rR, ...
    -cfgR.reward.scale*(iR.termVoltage+iR.termLoss+iR.termWear+iR.termInterp), 1e-10);
results = check(results,'the two forms differ (normalisation is not a no-op)', ...
    abs(rN - rR) > 1e-6);
results = checkNear(results,'raw form matches the legacy formula', rR, ...
    -cfgR.reward.scale*(cfgR.reward.alpha*iR.Vviol + cfgR.reward.beta*iR.Ploss + ...
     cfgR.reward.gamma*iR.actionChange + cfgR.reward.delta*iR.interp), 1e-10);

results = report(results);
end
