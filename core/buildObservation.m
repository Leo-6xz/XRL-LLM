function obs = buildObservation(state, cfg)
%BUILDOBSERVATION  Pack the environment state into the 80-element observation.
%
%   OBS = BUILDOBSERVATION(STATE, CFG)
%
%   LAYOUT (indices are fixed and mirrored by FEATUREMAPPING)
%   ---------------------------------------------------------
%      1 : 33   bus voltage magnitudes (p.u.)
%     34 : 65   active load at buses 2..33 (p.u.)
%     66 : 68   PV active output (p.u.)
%     69        aggregate load scale (dimensionless)
%     70        sin(2*pi*hour/24)
%     71        cos(2*pi*hour/24)
%     72 : 80   previous normalised action (9 elements)
%
%   WHY THE PREVIOUS ACTION IS IN THE OBSERVATION
%   ---------------------------------------------
%   Two of the reward terms -- device wear and the hysteresis component of
%   the interpretability penalty -- depend on the previous action. Without
%   it in the observation the problem is not Markov and PPO's value
%   function cannot be consistent. This is a small point that is easy to
%   miss and it materially affects convergence.
%
%   WHY REACTIVE LOAD IS NOT IN THE OBSERVATION
%   -------------------------------------------
%   Reactive load is reconstructed from active load through each bus's
%   nominal Q/P ratio (FEEDER.QPRATIO), so carrying it explicitly would be
%   32 redundant features. This keeps the vector at 80, matching the
%   "~70-80 features" specified in the brief. If you move to a scenario
%   generator with time-varying power factor, add the 32 channels back and
%   update CFG.DIMS.NOBS and FEATUREMAPPING together.
%
%   See also FEATUREMAPPING, XRLSTEP.

nPV = cfg.dims.nPV;

obs = zeros(cfg.dims.nObs, 1);

nB  = cfg.dims.nBus;              % 33 on the reference feeder

obs(1:nB)              = state.Vm(:);
obs(nB+1 : 2*nB-1)     = state.Pload(2:nB);
obs(2*nB : 2*nB-1+nPV) = state.pvP(:);

k = 2*nB - 1 + nPV;                % 68 on the 33-bus feeder
obs(k+1) = state.loadScale;
obs(k+2) = sin(2*pi*state.hour/24);
obs(k+3) = cos(2*pi*state.hour/24);
obs(k+4 : k+3+cfg.dims.nAction) = state.aPrev(:);

if any(~isfinite(obs))
    error('buildObservation:nonFinite', ...
        'Observation contains non-finite entries at indices %s.', ...
        mat2str(find(~isfinite(obs))'));
end
end
