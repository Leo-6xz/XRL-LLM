function results = test_environment()
%TEST_ENVIRONMENT  Validate reset/step, action decoding and the reward.

results = testSuite('environment');
cfg    = xrlConfig();
feeder = buildFeeder(cfg.net.dataFcn());
rs     = RandStream('mt19937ar','Seed',1);

% --- Dimensions -------------------------------------------------------
[obs, state] = xrlReset(cfg, feeder, rs);
results = check(results,'observation is 80 x 1', isequal(size(obs), [80 1]));
results = check(results,'observation finite',    all(isfinite(obs)));
results = check(results,'nAction == 9',          cfg.dims.nAction == 9);
results = check(results,'voltages in obs(1:33)', all(obs(1:33) > 0.5 & obs(1:33) < 1.5));

% --- Action decoding --------------------------------------------------
dev = decodeAction(zeros(9,1), cfg, state.pvP);
results = check(results,'neutral action -> tap 0',   dev.tap == 0);
results = check(results,'neutral -> nominal slack',  abs(dev.V0 - 1.0) < 1e-12);

devMax = decodeAction(ones(9,1), cfg, state.pvP);
results = check(results,'action +1 -> max tap',      devMax.tap == cfg.oltc.tapMax);
results = check(results,'action +1 -> all caps on',  all(devMax.capOn == 1));

devMin = decodeAction(-ones(9,1), cfg, state.pvP);
results = check(results,'action -1 -> min tap',      devMin.tap == cfg.oltc.tapMin);
results = check(results,'action -1 -> all caps off', all(devMin.capOn == 0));

results = check(results,'action clipped to [-1,1]', ...
    all(abs(decodeAction(5*ones(9,1), cfg, state.pvP).aClipped) <= 1));

% Inverter headroom must shrink as active output rises.
pvHigh = 0.99 * (cfg.pv.ratedKW(:)/1000/cfg.net.baseMVA) / cfg.pv.pfMin;
devHi  = decodeAction(ones(9,1), cfg, pvHigh);
devLo  = decodeAction(ones(9,1), cfg, zeros(cfg.dims.nPV,1));
results = check(results,'Q headroom shrinks at high PV', all(devHi.invQmax < devLo.invQmax));
results = check(results,'Q headroom non-negative',       all(devHi.invQmax >= 0));

% --- Sign convention --------------------------------------------------
pf0 = solveWithDevices(state, decodeAction(zeros(9,1), cfg, state.pvP), cfg, feeder);
aCap = zeros(9,1); aCap(1+find(cfg.cap.buses==30,1)) = 1;
pfC  = solveWithDevices(state, decodeAction(aCap, cfg, state.pvP), cfg, feeder);
results = check(results,'closing a bank raises local voltage', pfC.Vm(30) > pf0.Vm(30));

aTap = zeros(9,1); aTap(1) = 1;
pfT  = solveWithDevices(state, decodeAction(aTap, cfg, state.pvP), cfg, feeder);
results = check(results,'raising the tap raises voltages', all(pfT.Vm >= pf0.Vm - 1e-12));

% --- Episode rollout --------------------------------------------------
[obs, state] = xrlReset(cfg, feeder, rs);
nSteps = 0; done = false; rewards = [];
while ~done
    a = 0.2*randn(9,1);
    [obs, r, done, state, info] = xrlStep(a, state, cfg, feeder);
    rewards(end+1) = r; %#ok<AGROW>
    nSteps = nSteps + 1;
    if nSteps > 100, break; end
end
results = check(results,'episode length matches config', nSteps == cfg.episode.stepsPerEpisode);
results = check(results,'rewards finite',    all(isfinite(rewards)));
results = check(results,'rewards non-positive', all(rewards <= 0));
results = check(results,'observation stays finite', all(isfinite(obs)));
results = check(results,'history logged',    numel(state.history) == nSteps);
results = check(results,'info has VmPre/VmPost', ...
    isfield(info,'VmPre') && isfield(info,'VmPost'));

% --- Determinism ------------------------------------------------------
r1 = rollout(cfg, feeder, 11);
r2 = rollout(cfg, feeder, 11);
results = checkNear(results,'same seed -> same rollout', max(abs(r1-r2)), 0, 1e-12);
r3 = rollout(cfg, feeder, 12);
results = check(results,'different seed -> different rollout', max(abs(r1-r3)) > 1e-9);

results = report(results);
end

function rr = rollout(cfg, feeder, seed)
rs = RandStream('mt19937ar','Seed',seed);
[~, state] = xrlReset(cfg, feeder, rs);
rr = []; done = false; k = 0;
while ~done && k < 50
    [~, r, done, state] = xrlStep(zeros(9,1), state, cfg, feeder);
    rr(end+1) = r; %#ok<AGROW>
    k = k + 1;
end
rr = rr(:);
end
