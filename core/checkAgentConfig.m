function [ok, diffs] = checkAgentConfig(savedCfg, cfg, opts)
%CHECKAGENTCONFIG  Detect train/serve mismatch on a reloaded agent.
%
%   [OK, DIFFS] = CHECKAGENTCONFIG(SAVEDCFG, CFG)
%
%   Compares the configuration an agent was TRAINED under against the one
%   about to be used to EVALUATE and EXPLAIN it, and reports any field that
%   changes the environment's dynamics or its reward.
%
%   WHY THIS MATTERS MORE THAN IT LOOKS
%   -----------------------------------
%   A reward-weight change does not break anything loudly. The agent still
%   runs, the rollouts still complete, the numbers still print. But every
%   reward-derived quantity -- mean reward, the interpretability penalty
%   logged with each SHAP record, the reward decomposition --
%   now describes a reward the policy never optimized. That is a silent
%   train/serve mismatch, and it is the kind of error that survives review
%   because nothing in the output looks wrong.
%
%   This project switched from raw reward weights to a scale-free
%   normalized form partway through. Any agent trained before that switch
%   must be evaluated with cfg.reward.normalize = false to reproduce the
%   figures it was trained against.
%
%   Fields checked: everything that alters the environment (device
%   placement and ratings, limits, episode structure, scenario ranges,
%   observation/action dimensions) and everything that alters the reward.
%
%   See also MAIN, TRAINPPO, REWARDBALANCE.

if nargin < 3 || isempty(opts), opts = struct(); end
if ~isfield(opts,'verbose'), opts.verbose = true; end

diffs = {};

% --- Reward definition ------------------------------------------------
sN = getfielddef(savedCfg.reward, 'normalize', false);
cN = getfielddef(cfg.reward,      'normalize', false);
if sN ~= cN
    diffs{end+1} = sprintf('reward.normalize: trained %d, now %d', sN, cN);
end
if sN && cN
    diffs = cmpStruct(diffs, savedCfg.reward.w,   cfg.reward.w,   'reward.w');
    diffs = cmpStruct(diffs, savedCfg.reward.ref, cfg.reward.ref, 'reward.ref');
else
    for f = {'alpha','beta','gamma','delta','scale'}
        diffs = cmpScalar(diffs, savedCfg.reward, cfg.reward, f{1}, 'reward');
    end
end
diffs = cmpStruct(diffs, savedCfg.interp, cfg.interp, 'interp');

% --- Environment ------------------------------------------------------
diffs = cmpStruct(diffs, savedCfg.limits,  cfg.limits,  'limits');
diffs = cmpStruct(diffs, savedCfg.oltc,    cfg.oltc,    'oltc');
diffs = cmpStruct(diffs, savedCfg.episode, cfg.episode, 'episode');
diffs = cmpVec(diffs, savedCfg.cap.buses,   cfg.cap.buses,   'cap.buses');
diffs = cmpVec(diffs, savedCfg.cap.kvar,    cfg.cap.kvar,    'cap.kvar');
diffs = cmpVec(diffs, savedCfg.pv.buses,    cfg.pv.buses,    'pv.buses');
diffs = cmpVec(diffs, savedCfg.pv.ratedKW,  cfg.pv.ratedKW,  'pv.ratedKW');
diffs = cmpStruct(diffs, savedCfg.dims,     cfg.dims,        'dims');

ok = isempty(diffs);

if opts.verbose && ~ok
    fprintf('\n  !! AGENT CONFIGURATION MISMATCH (%d field(s)) !!\n', numel(diffs));
    for i = 1:numel(diffs)
        fprintf('     %s\n', diffs{i});
    end
    fprintf(['     The agent was trained under a different setup. Every\n' ...
             '     reward-derived number below describes a reward this policy\n' ...
             '     never optimized. Either load the matching config with\n' ...
             '        S = load(cfg.train.savePath); cfg = S.cfg;\n' ...
             '     or retrain under the current one.\n\n']);
end
end

% =====================================================================
function d = cmpStruct(d, a, b, prefix)
if ~isstruct(a) || ~isstruct(b), return; end
f = union(fieldnames(a), fieldnames(b));
for i = 1:numel(f)
    d = cmpScalar(d, a, b, f{i}, prefix);
end
end

function d = cmpScalar(d, a, b, name, prefix)
if ~isfield(a,name) || ~isfield(b,name), return; end
va = a.(name); vb = b.(name);
if ~isnumeric(va) || ~isnumeric(vb) || ~isequal(size(va), size(vb))
    if ~isequaln(va, vb)
        d{end+1} = sprintf('%s.%s changed', prefix, name); %#ok<AGROW>
    end
    return
end
if any(abs(va(:) - vb(:)) > 1e-12)
    d{end+1} = sprintf('%s.%s: trained %s, now %s', prefix, name, ...
        mat2str(va,4), mat2str(vb,4)); %#ok<AGROW>
end
end

function d = cmpVec(d, a, b, name)
if ~isequal(size(a), size(b)) || any(abs(a(:)-b(:)) > 1e-12)
    d{end+1} = sprintf('%s: trained %s, now %s', name, mat2str(a), mat2str(b)); %#ok<AGROW>
end
end

function v = getfielddef(s, f, dflt)
if isfield(s, f), v = s.(f); else, v = dflt; end
end
