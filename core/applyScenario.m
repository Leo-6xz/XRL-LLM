function state = applyScenario(state, cfg, feeder, t)
%APPLYSCENARIO  Set loads, PV output and clock for step T from the profile.
%
%   STATE = APPLYSCENARIO(STATE, CFG, FEEDER, T)
%
%   Reactive load is derived from active load through each bus's nominal
%   Q/P ratio, which is why the observation vector carries only active
%   load (see BUILDOBSERVATION).
%
%   See also XRLRESET, XRLSTEP, LOADPROFILES.

prof = state.prof;

if t < 1 || t > prof.T
    error('applyScenario:badStep', 'Step %d outside episode length %d.', t, prof.T);
end

state.t         = t;
state.hour      = prof.hour(t);
state.loadScale = prof.loadScale(t);

state.Pload = feeder.Pload0 .* prof.busScale(:, t);
state.Qload = state.Pload .* feeder.qpRatio;

pvRatedPU = cfg.pv.ratedKW(:) / 1000 / cfg.net.baseMVA;
state.pvP = pvRatedPU .* prof.pvFrac(:, t);
end
