function S = voltageSensitivity(feeder, state, cfg, dev)
%VOLTAGESENSITIVITY  Numerical dV/dQ and dV/dTap at the current operating point.
%
%   S = VOLTAGESENSITIVITY(FEEDER, STATE, CFG, DEV) perturbs each
%   controllable reactive source in turn and re-solves the power flow,
%   returning
%
%     S.dVdQ  [nBus x nDevQ] volts (p.u.) per p.u. of injected VAr
%     S.dVdT  [nBus x 1]     volts (p.u.) per OLTC tap step
%     S.devBus[1 x nDevQ]    the bus of each reactive device
%     S.devName{1 x nDevQ}
%
%   USED FOR TWO THINGS
%   -------------------
%   1. The LLM prompt quotes the predicted voltage change so the
%      explanation's "expected effect" clause is a physical claim that can
%      be checked, not a plausible-sounding guess. This is the difference
%      between an explanation and a rationalisation.
%   2. COUNTERFACTUALTEST uses it to sanity-check that the direction the
%      explanation asserts matches the direction the network delivers.
%
%   Cost is (nDevQ + 1) power-flow solves, i.e. sub-millisecond here, so
%   it can be recomputed at every explained decision rather than assumed
%   constant. That matters -- sensitivities on a distribution feeder vary
%   substantially between light and heavy load.
%
%   See also RADIALPOWERFLOW, GENERATEEXPLANATION.

if nargin < 4 || isempty(dev)
    dev = decodeAction(zeros(cfg.dims.nAction,1), cfg, state.pvP);
end

nCap = cfg.dims.nCap;
nPV  = cfg.dims.nPV;
nDev = nCap + nPV;

% Base solve
[pf0, Pinj, Qinj] = solveWithDevices(state, dev, cfg, feeder);
V0m = pf0.Vm;

% Perturbation size: 50 kvar. Small enough to stay linear, large enough
% to sit far above the power-flow convergence tolerance (1e-8).
dQ = 0.050 / cfg.net.baseMVA;

devBus  = [cfg.cap.buses(:); cfg.pv.buses(:)]';
devName = cell(1, nDev);
for k = 1:nCap
    devName{k} = sprintf('Capacitor bank bus %d', cfg.cap.buses(k));
end
for k = 1:nPV
    devName{nCap+k} = sprintf('PV inverter bus %d', cfg.pv.buses(k));
end

dVdQ = zeros(feeder.nBus, nDev);
for k = 1:nDev
    b = devBus(k);
    Q2 = Qinj;
    Q2(b) = Q2(b) - dQ;                  % inject VArs -> reduce consumed Q
    pfk = radialPowerFlow(feeder, Pinj, Q2, dev.V0);
    dVdQ(:,k) = (pfk.Vm - V0m) / dQ;
end

% OLTC sensitivity
pfT = radialPowerFlow(feeder, Pinj, Qinj, dev.V0 + cfg.oltc.stepPU);
dVdT = (pfT.Vm - V0m) / 1;               % per tap step

S.dVdQ    = dVdQ;
S.dVdT    = dVdT;
S.devBus  = devBus;
S.devName = devName;
S.Vbase   = V0m;
end
