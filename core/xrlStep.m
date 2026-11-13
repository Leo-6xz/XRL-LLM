function [obs, reward, isDone, state, info] = xrlStep(action, state, cfg, feeder)
%XRLSTEP  Advance the VVC environment by one control interval.
%
%   [OBS, REWARD, ISDONE, STATE, INFO] = XRLSTEP(ACTION, STATE, CFG, FEEDER)
%
%   Order of operations (this matters for the explanation layer):
%     1. Decode ACTION against the *current* PV output, so the inverter
%        reactive limits reflect the conditions the agent observed.
%     2. Solve the power flow with the new device positions at the
%        *current* load level. This is the immediate physical consequence
%        of the action and is what the reward scores.
%     3. Advance the clock, apply the next step's load/PV, and re-solve to
%        produce the observation the agent will act on next.
%
%   Splitting steps 2 and 3 keeps the credit assignment clean: the reward
%   measures the action's effect, not the arrival of new load.
%
%   INFO carries the full reward decomposition plus the pre-action and
%   post-action voltage profiles, which the SHAP records and the LLM
%   prompt both consume.
%
%   See also XRLRESET, COMPUTEREWARD, DECODEACTION.

action = action(:);

% ---- 1. Decode ------------------------------------------------------
dev = decodeAction(action, cfg, state.pvP);

% Attach the previous move direction so the hysteresis term of the
% interpretability penalty has something to compare against.
if ~isempty(state.devPrev) && isfield(state, 'lastDQ')
    state.devPrev.dQ = state.lastDQ;
end

% ---- 2. Immediate physical consequence ------------------------------
preState = state;                      % pre-action snapshot for the penalty
[pf, Pinj, Qinj] = solveWithDevices(state, dev, cfg, feeder);

[reward, info] = computeReward(preState, pf, dev, state.devPrev, feeder, cfg);

info.VmPre    = preState.Vm;
info.VmPost   = pf.Vm;
info.hour     = state.hour;
info.t        = state.t;
info.scenario = state.prof.scenarioTag;
info.dev      = dev;
info.pf       = pf;

% ---- Log ------------------------------------------------------------
h.t      = state.t;
h.action = action;
h.dev    = dev;
h.reward = reward;
h.info   = info;
h.Vm     = pf.Vm;
h.obs    = buildObservation(preState, cfg);
state.history(end+1) = h;

% ---- 3. Advance the clock -------------------------------------------
tNext  = state.t + 1;
isDone = tNext > state.prof.T;

state.aPrev   = dev.aClipped;
state.devPrev = dev;
state.lastDQ  = info.interpBreakdown.dQ;

if isDone
    % Terminal: return the post-action observation so callers that read
    % OBS after termination still get a valid, finite vector.
    state.pf   = pf;
    state.V    = pf.V;
    state.Vm   = pf.Vm;
    state.Pinj = Pinj;
    state.Qinj = Qinj;
    state.dev  = dev;
    obs = buildObservation(state, cfg);
    return
end

state = applyScenario(state, cfg, feeder, tNext);

% Re-solve at the new load level, holding the devices where the agent
% left them. This is the state the agent will observe next.
[pfNext, PinjNext, QinjNext] = solveWithDevices(state, dev, cfg, feeder);

state.pf   = pfNext;
state.V    = pfNext.V;
state.Vm   = pfNext.Vm;
state.Pinj = PinjNext;
state.Qinj = QinjNext;
state.dev  = dev;

obs = buildObservation(state, cfg);
end
