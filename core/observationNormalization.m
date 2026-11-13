function norm = observationNormalization(cfg, feeder)
%OBSERVATIONNORMALIZATION  Fixed scale/bias to condition the 80-D observation.
%
%   NORM = OBSERVATIONNORMALIZATION(CFG, FEEDER) returns column vectors
%   NORM.SCALE and NORM.BIAS such that
%
%       zNormalised = SCALE .* z + BIAS
%
%   maps every observation channel to roughly zero mean and unit range.
%   Used as a SCALINGLAYER immediately after the input of both the actor
%   and the critic (see CREATEPPOAGENT).
%
%   WHY THIS IS NEEDED
%   ------------------
%   The raw observation mixes quantities spanning three orders of
%   magnitude: bus voltages sit near 1.0 with a useful range of about
%   +-0.05, while per-bus active loads range from 0.0045 to 0.042 p.u.
%   A single dense layer over that mixture is badly conditioned -- the
%   voltage channels dominate the first-layer gradients and the load
%   channels are effectively invisible, so the policy cannot learn to
%   anticipate load.
%
%   WHY FIXED CONSTANTS RATHER THAN RUNNING STATISTICS
%   --------------------------------------------------
%   MATLAB's featureInputLayer supports 'zscore' normalization, but it
%   needs mean/std computed from a training set, which would have to be
%   collected, stored with the agent, and reapplied identically at
%   explanation time. Every one of those is a chance for train/serve skew.
%   The bounds here are known from the network data and the scenario
%   generator, so the mapping is deterministic, reproducible, and
%   identical in training, SHAP and the dashboard.
%
%   IMPORTANT: because normalization lives INSIDE the network, POLICYFCN
%   and the SHAP layer still operate on RAW observations. Do not normalise
%   twice.
%
%   See also CREATEPPOAGENT, BUILDOBSERVATION.

if nargin < 1 || isempty(cfg),    cfg = xrlConfig(); end
if nargin < 2 || isempty(feeder), feeder = buildFeeder(cfg.net.dataFcn()); end

nObs = cfg.dims.nObs;
nPV  = cfg.dims.nPV;
nA   = cfg.dims.nAction;

centre = zeros(nObs,1);
spread = ones(nObs,1);

% --- 1:33 bus voltages: centred on nominal, scaled by the statutory band
nB = cfg.dims.nBus;                 % 33 on the reference feeder
centre(1:nB) = cfg.limits.Vnom;
spread(1:nB) = (cfg.limits.Vmax - cfg.limits.Vmin) / 2;    % 0.05 p.u.

% --- 34:65 active load at buses 2..33: per-bus, from the nominal profile
%     Each bus is centred on its own mid-range demand and scaled by the
%     spread the scenario generator can produce. Using one global scale
%     here would leave the small buses at ~0.1 and the large ones at ~1.
r  = cfg.episode.loadScaleRange;
mid  = mean(r);
half = (r(2) - r(1)) / 2;
P0 = feeder.Pload0(2:nB);
centre(nB+1:2*nB-1) = P0 * mid;
spread(nB+1:2*nB-1) = max(P0 * half, 1e-4);

% --- 66:68 PV active output
pvRatedPU = cfg.pv.ratedKW(:) / 1000 / cfg.net.baseMVA;
centre(2*nB:2*nB-1+nPV) = pvRatedPU / 2;
spread(2*nB:2*nB-1+nPV) = max(pvRatedPU / 2, 1e-6);

k = 2*nB - 1 + nPV;

% --- aggregate load level
centre(k+1) = mid;
spread(k+1) = max(half, 1e-6);

% --- time of day (already on [-1,1]) and previous action (already [-1,1])
centre(k+2 : k+3) = 0;   spread(k+2 : k+3) = 1;
centre(k+4 : k+3+nA) = 0; spread(k+4 : k+3+nA) = 1;

norm.scale = 1 ./ spread;
norm.bias  = -centre ./ spread;

assert(numel(norm.scale) == nObs && numel(norm.bias) == nObs, ...
    'observationNormalization:size', 'Normalisation vectors must be %d long.', nObs);
assert(all(isfinite(norm.scale)) && all(isfinite(norm.bias)), ...
    'observationNormalization:nonFinite', 'Normalisation contains non-finite values.');
end
