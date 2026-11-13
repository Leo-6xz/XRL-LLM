function [obs, state] = xrlReset(cfg, feeder, rngStream)
%XRLRESET  Start a new VVC episode.
%
%   [OBS, STATE] = XRLRESET(CFG, FEEDER) draws a fresh 24-hour scenario,
%   solves the power flow at t = 0 with all devices at their neutral
%   position, and returns the initial observation plus the full internal
%   state struct.
%
%   The STATE struct is the environment's memory. It is deliberately a
%   plain struct with no toolbox dependency so that SHAP, the LLM layer,
%   the dashboard and the unit tests can all drive the environment on a
%   base MATLAB installation.
%
%   STATE fields
%   ------------
%     .t, .hour, .prof         scenario and clock
%     .Vm, .V, .pf             latest power-flow solution
%     .Pload, .Qload           current bus loads (p.u.)
%     .pvP                     current PV active output (p.u.)
%     .loadScale               aggregate load multiplier now
%     .aPrev                   previous normalized action
%     .devPrev                 previous decoded device struct
%     .history                 per-step log (populated by XRLSTEP)
%
%   See also XRLSTEP, LOADPROFILES.

if nargin < 3 || isempty(rngStream)
    rngStream = RandStream.getGlobalStream();
end

prof = loadProfiles(cfg, rngStream);

state.cfgHash   = cfg.dims;         % cheap guard against config drift
state.t         = 1;
state.prof      = prof;
state.hour      = prof.hour(1);
state.loadScale = prof.loadScale(1);

% Neutral starting position: tap 0, all banks open, inverters at unity PF.
state.aPrev   = zeros(cfg.dims.nAction, 1);
state.devPrev = [];

state = applyScenario(state, cfg, feeder, 1);

% Solve the power flow with the neutral device position so the agent's
% first observation reflects an uncontrolled feeder.
dev0 = decodeAction(state.aPrev, cfg, state.pvP);
[pf, Pinj, Qinj] = solveWithDevices(state, dev0, cfg, feeder);

state.pf    = pf;
state.V     = pf.V;
state.Vm    = pf.Vm;
state.Pinj  = Pinj;
state.Qinj  = Qinj;
state.dev   = dev0;

state.history = struct('t',{},'action',{},'dev',{},'reward',{}, ...
                       'info',{},'Vm',{},'obs',{});

obs = buildObservation(state, cfg);
end
