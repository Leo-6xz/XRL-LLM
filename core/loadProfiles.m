function prof = loadProfiles(cfg, rngStream)
%LOADPROFILES  Generate one 24-hour operating scenario.
%
%   PROF = LOADPROFILES(CFG) draws a random daily scenario: an aggregate
%   load shape, a PV clear-sky fraction, per-bus load noise, and cloud
%   transients. Each episode of training sees a fresh draw, which is what
%   forces the policy to generalize instead of memorizing one day.
%
%   PROF = LOADPROFILES(CFG, RNGSTREAM) uses a supplied RandStream for
%   reproducibility.
%
%   Output PROF
%   -----------
%     .T          number of steps (= cfg.episode.stepsPerEpisode)
%     .hour       [1 x T] hour of day
%     .loadScale  [1 x T] aggregate load multiplier
%     .busScale   [nBus x T] per-bus multiplier (loadScale + noise)
%     .pvFrac     [nPV x T] PV output as a fraction of rated
%     .peakScale  scalar daily peak multiplier that was drawn
%     .pvPeak     scalar clear-sky fraction that was drawn
%     .scenarioTag one of 'lightLoad','heavyLoad','highPV','reversePower',
%                  'nominal' -- used to stratify the SHAP and G-Eval samples
%
%   The shapes are deliberately simple and analytic rather than read from
%   a data file, so the repository is self-contained. Swap in measured
%   feeder data by replacing this function; nothing else needs to change.

if nargin < 2 || isempty(rngStream)
    rngStream = RandStream.getGlobalStream();
end

T    = cfg.episode.stepsPerEpisode;
nBus = cfg.dims.nBus;
nPV  = cfg.dims.nPV;

hour = (0:T-1) * (24 / T);

% ---------------------------------------------------------------------
% Aggregate load: two-peak residential/commercial shape (morning and
% evening), normalised so the maximum of the shape is 1.
% ---------------------------------------------------------------------
shape = 0.55 ...
      + 0.25 * exp(-0.5*((hour - 8 )/2.0).^2) ...
      + 0.45 * exp(-0.5*((hour - 19)/2.5).^2);
shape = shape / max(shape);

r = cfg.episode.loadScaleRange;
peakScale = r(1) + (r(2)-r(1)) * rand(rngStream);
loadScale = peakScale * shape;

% ---------------------------------------------------------------------
% Per-bus multiplicative noise (diversity between feeders / customers)
% ---------------------------------------------------------------------
busNoise = 1 + cfg.episode.loadNoiseStd * randn(rngStream, nBus, T);
busNoise = max(0.5, busNoise);
busScale = repmat(loadScale, nBus, 1) .* busNoise;

% ---------------------------------------------------------------------
% PV: clear-sky bell between sunrise and sunset, scaled by a daily
% cloudiness draw, plus per-inverter cloud transients.
% ---------------------------------------------------------------------
sunrise = 6; sunset = 19;
clear = zeros(1, T);
day   = hour >= sunrise & hour <= sunset;
clear(day) = sin(pi * (hour(day) - sunrise) / (sunset - sunrise));
clear = max(0, clear);

p = cfg.episode.pvPeakRange;
pvPeak = p(1) + (p(2)-p(1)) * rand(rngStream);

pvNoise = 1 + cfg.episode.pvNoiseStd * randn(rngStream, nPV, T);
pvFrac  = pvPeak * repmat(clear, nPV, 1) .* pvNoise;
pvFrac  = max(0, min(1, pvFrac));

% ---------------------------------------------------------------------
% Label the scenario. The tag is not used by the environment; it exists
% so the evaluation code can report metrics per operating regime, which
% is exactly the "normal vs. stressed" stratification that reviewers of
% XAI-for-power-systems papers ask for.
% ---------------------------------------------------------------------
totalPVpu  = pvPeak * sum(cfg.pv.ratedKW) / 1000 / cfg.net.baseMVA;
totalLoadMW = 3.715;                       % IEEE 33-bus nominal demand
if isfield(cfg.net, 'totalLoadMW'), totalLoadMW = cfg.net.totalLoadMW; end
totalLdpu  = peakScale * totalLoadMW / cfg.net.baseMVA;
if totalPVpu > totalLdpu
    tag = 'reversePower';
elseif pvPeak > 0.7
    tag = 'highPV';
elseif peakScale > 0.95
    tag = 'heavyLoad';
elseif peakScale < 0.65
    tag = 'lightLoad';
else
    tag = 'nominal';
end

prof.T          = T;
prof.hour       = hour;
prof.loadScale  = loadScale;
prof.busScale   = busScale;
prof.pvFrac     = pvFrac;
prof.peakScale  = peakScale;
prof.pvPeak     = pvPeak;
prof.scenarioTag= tag;
end