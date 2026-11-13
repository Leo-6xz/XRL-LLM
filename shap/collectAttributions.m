function [records, meta] = collectAttributions(policyFcn, cfg, feeder, nEpisodes, opts)
%COLLECTATTRIBUTIONS  Build the attribution-action record set.
%
%   [RECORDS, META] = COLLECTATTRIBUTIONS(POLICYFCN, CFG, FEEDER, NEPISODES)
%
%   Rolls the policy out, computes a SHAP explanation at every step that
%   involved an actual device movement, and returns the structured records
%   that Step 3.3 of the brief specifies.
%
%   RECORD fields
%   -------------
%     .timestamp .episode .t .hour .scenario
%     .state       full internal state snapshot (for counterfactuals)
%     .obs         [nObs x 1] observation the policy acted on
%     .action      [nAction x 1] normalised action
%     .dev         decoded device struct
%     .actionText  human sentence from DESCRIBEACTION
%     .actionBrief condensed one-clause form, for the summary level
%     .phi         [1 x M] SHAP values
%     .playerNames {1 x M}
%     .topFeatures struct array from FINDTOPFEATURES
%     .shapMeta    which action channel was explained, method, phi0
%     .reward .info
%     .sensitivity dV/dQ at this operating point
%
%   See also EXACTSHAP, KERNELSHAP, GENERATEEXPLANATION.

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'explainAll'),  opts.explainAll  = false; end
if ~isfield(opts,'maxRecords'),  opts.maxRecords  = 200;   end
if ~isfield(opts,'verbose'),     opts.verbose     = cfg.io.verbose; end
if ~isfield(opts,'seed'),        opts.seed        = cfg.seed; end

map = featureMapping(cfg);

% ---------------------------------------------------------------------
% Pass 1: roll out to gather the background distribution AND the steps to
% explain. Doing it in two passes means the background is representative
% of the whole run rather than only its beginning.
% ---------------------------------------------------------------------
rs = RandStream('mt19937ar','Seed',opts.seed);

allObs   = [];
pending  = struct('episode',{},'t',{},'obs',{},'action',{},'dev',{}, ...
                  'devPrev',{},'state',{},'reward',{},'info',{});

for ep = 1:nEpisodes
    [obs, state] = xrlReset(cfg, feeder, rs);
    done = false;
    while ~done
        preState = state;
        devPrev  = state.devPrev;
        a = policyFcn(obs(:)');
        a = a(:);

        allObs(end+1,:) = obs(:)'; %#ok<AGROW>

        [obs, reward, done, state, info] = xrlStep(a, state, cfg, feeder);

        moved = info.interpBreakdown.nMoved > 0;
        if opts.explainAll || moved
            p = struct();
            p.episode = ep;
            p.t       = info.t;
            p.obs     = preState_obs(preState, cfg);
            p.action  = a;
            p.dev     = info.dev;
            p.devPrev = devPrev;
            p.state   = preState;
            p.reward  = reward;
            p.info    = info;
            pending(end+1) = p; %#ok<AGROW>
        end
    end
end

if isempty(pending)
    warning('collectAttributions:noMovement', ...
        ['The policy never moved a device across %d episodes, so there is ', ...
         'nothing to explain. This normally means the agent is undertrained ', ...
         'or the interpretability weight cfg.reward.delta is so high that ', ...
         'inaction dominates. Re-run with opts.explainAll = true to inspect.'], ...
        nEpisodes);
    records = struct([]);
    meta = struct('nCandidates',0);
    return
end

% Subsample if there are more candidates than the SHAP budget allows.
nCand = numel(pending);
if nCand > opts.maxRecords
    sel = round(linspace(1, nCand, opts.maxRecords));
    pending = pending(sel);
end

% ---------------------------------------------------------------------
% Background: stratified sample of observed states
% ---------------------------------------------------------------------
nBG = min(cfg.shap.nBackground, size(allObs,1));
bgIdx = round(linspace(1, size(allObs,1), nBG));
background = allObs(bgIdx, :);

% ---------------------------------------------------------------------
% Choose players (groups) according to the configured SHAP mode
% ---------------------------------------------------------------------
[groups, playerNames, groupingTag] = shapGroups(cfg, map);
if opts.verbose
    fprintf('  SHAP grouping: %s (%d players)\n', groupingTag, numel(groups));
end
if strcmpi(groupingTag, 'raw')
    engine = @kernelSHAP;
    sopts = struct('nCoalitions', cfg.shap.nCoalitions, ...
                   'batchSize', cfg.shap.batchSize, ...
                   'verbose', false, 'seed', cfg.seed);
else
    engine = @exactSHAP;
    sopts = struct('maxExactM', cfg.shap.maxExactM, ...
                   'batchSize', cfg.shap.batchSize, 'verbose', false);
end

% ---------------------------------------------------------------------
% Pass 2: explain each retained decision
% ---------------------------------------------------------------------
records = struct([]);
tStart  = tic;

for i = 1:numel(pending)
    p = pending(i);

    % Which action channel to explain: the one that moved most, unless the
    % user pinned a channel in the config.
    if isnumeric(cfg.shap.targetAction)
        k = cfg.shap.targetAction;
    else
        dq = abs(p.info.interpBreakdown.dQ);
        [~, k] = max(dq);
    end

    [predictFcn, sMeta] = shapValueFunction(policyFcn, k, cfg);
    [phi, sOut] = engine(predictFcn, p.obs(:)', background, groups, sopts);

    r = struct();
    r.timestamp   = datetime('now');
    r.episode     = p.episode;
    r.t           = p.t;
    r.hour        = p.info.hour;
    r.scenario    = p.info.scenario;
    r.state       = p.state;
    r.obs         = p.obs(:);
    r.action      = p.action;
    r.dev         = p.dev;
    r.devPrev     = p.devPrev;
    [r.actionText, ~, r.actionBrief] = describeAction(p.dev, p.devPrev, cfg);
    r.phi         = phi;
    r.playerNames = playerNames;
    nf = 0;
    if isfield(sOut,'noiseFloor'), nf = sOut.noiseFloor; end
    r.topFeatures = findTopFeatures(phi, cfg.shap.topK, playerNames, nf);
    r.shapMeta    = sOut;
    r.shapMeta.explainedChannel     = k;
    r.shapMeta.explainedChannelName = sMeta.name;
    r.reward      = p.reward;
    r.info        = p.info;
    r.sensitivity = voltageSensitivity(feeder, p.state, cfg, p.dev);

    if isempty(records)
        records = r;
    else
        records(end+1) = r; %#ok<AGROW>
    end

    if opts.verbose && mod(i, 10) == 0
        fprintf('  SHAP: %d/%d explained (%.1f s elapsed)\n', ...
            i, numel(pending), toc(tStart));
    end
end

meta.nCandidates    = nCand;
meta.nExplained     = numel(records);
meta.background     = background;
meta.groups         = groups;
meta.playerNames    = playerNames;
meta.method         = sOut.method;
meta.elapsedSec     = toc(tStart);
meta.secPerRecord   = meta.elapsedSec / max(1, numel(records));
end

% =====================================================================
function o = preState_obs(preState, cfg)
%PRESTATE_OBS  Rebuild the observation the agent saw at this step.
o = buildObservation(preState, cfg);
end
