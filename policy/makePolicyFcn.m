function policyFcn = makePolicyFcn(agent, cfg)
%MAKEPOLICYFCN  Adapt any policy into the batched, deterministic form SHAP needs.
%
%   POLICYFCN = MAKEPOLICYFCN(AGENT, CFG) returns a handle with signature
%
%       A = policyFcn(X)     X: [n x nObs]   ->   A: [n x nAction]
%
%   AGENT may be
%     * an rlPPOAgent            -> uses the actor's DETERMINISTIC MEAN
%     * a function handle        -> used directly (must already be batched)
%     * [] or 'rule'             -> the rule-based baseline RULEPOLICY
%
%   WHY THE MEAN AND NOT GETACTION
%   ------------------------------
%   GETACTION on a PPO agent SAMPLES from the Gaussian policy. Two calls
%   with identical inputs return different actions. Feeding that to SHAP
%   is fatal: the Shapley efficiency axiom (sum of attributions equals
%   f(x) - E[f]) is violated by an amount driven purely by sampling noise,
%   and the resulting "explanations" are largely noise. This function
%   therefore calls EVALUATE on the extracted actor and keeps only the
%   mean head, which is deterministic.
%
%   The same argument applies to deployment: the setpoint an operator is
%   shown should be the policy mean, not a random draw. Exploration
%   belongs in training only.
%
%   WHY BATCHING MATTERS
%   --------------------
%   A single grouped SHAP explanation evaluates the policy on
%   2^12 x 64 = 262144 observation rows. Looping one row at a time through
%   GETACTION takes minutes per explanation; the batched EVALUATE call
%   takes well under a second. If you supply your own handle, make it
%   accept a matrix.
%
%   See also RULEPOLICY, SHAPVALUEFUNCTION, COLLECTATTRIBUTIONS.

if nargin < 2 || isempty(cfg)
    cfg = xrlConfig();
end

% ---- Function handle supplied directly -------------------------------
if isa(agent, 'function_handle')
    policyFcn = @(X) validateOut(agent(X), size(X,1), cfg);
    return
end

% ---- Rule-based baseline ---------------------------------------------
if isempty(agent) || (ischar(agent) && strcmpi(agent,'rule')) || ...
        (isstring(agent) && strcmpi(agent,"rule"))
    feeder = buildFeeder(cfg.net.dataFcn());
    policyFcn = @(X) rulePolicy(X, cfg, feeder);
    return
end

% ---- RL Toolbox agent -------------------------------------------------
actor = [];
try
    actor = getActor(agent);
catch
    % fall through to the getAction path below
end

if ~isempty(actor)
    policyFcn = @(X) actorMean(actor, X, cfg);
    % Self-test on a BATCH before returning. A single-row probe would pass
    % even with the N-D shape bug above, so the probe must use >= 2 rows.
    % This turns a crash deep inside the SHAP stage -- minutes into a run --
    % into an immediate, located error.
    probeBatch(policyFcn, cfg);
    return
end

warning('makePolicyFcn:fallback', ...
    ['Could not extract an actor from the supplied agent; falling back to ', ...
     'per-row getAction. SHAP will be slow and, if the policy is ', ...
     'stochastic, the attributions will not satisfy the efficiency axiom.']);
policyFcn = @(X) slowGetAction(agent, X, cfg);
end

% =====================================================================
function A = actorMean(actor, X, cfg)
%ACTORMEAN  Deterministic mean of a Gaussian actor, evaluated on a batch.
%
%   MATLAB approximators take observations as [obsDim x batchSize], which
%   is the transpose of the row-major convention used everywhere else in
%   this project, hence the X'.
n = size(X,1);
obsBatch = X';                                  % [nObs x n]

out = evaluate(actor, {obsBatch});

% A Gaussian actor returns {mean, standardDeviation}; a deterministic
% actor returns {action}. Either way the first output is what we want.
A = double(extractdata_safe(out{1}));

% SHAPE HANDLING -- THIS IS WHERE IT BREAKS IF YOU ASSUME 2-D.
% MATLAB approximators return [outputDim x batchSize x sequenceLength].
% For a SINGLE observation the trailing singletons collapse and the result
% is 2-D, so a plain transpose works and everything looks fine. For a
% BATCH the result is genuinely 3-D ([nAction x 1 x n]) and the transpose
% throws "TRANSPOSE does not support N-D arrays".
%
% That asymmetry is why the failure surfaced only in the SHAP stage: the
% environment rollouts call the policy one observation at a time and pass,
% then the first batched coalition evaluation crashes minutes later.
% Collapsing every trailing dimension into the batch dimension handles the
% 2-D and 3-D cases identically.
A = reshape(A, size(A,1), []);                   % -> [nAction x n]
A = A';                                          % -> [n x nAction]

A = max(-1, min(1, A));
A = validateOut(A, n, cfg);
end

% =====================================================================
function probeBatch(policyFcn, cfg)
%PROBEBATCH  Verify the policy handles a multi-row input correctly.
nProbe = 3;
X = repmat(0.5*ones(1, cfg.dims.nObs), nProbe, 1);
X(:,1:cfg.dims.nBus) = 1.0;                                  % plausible voltages
try
    A = policyFcn(X);
catch ME
    error('makePolicyFcn:batchFailed', ...
        ['The policy could not evaluate a %d-row batch: %s\n' ...
         'SHAP evaluates the policy on thousands of rows at a time, so a ', ...
         'policy that only accepts a single observation cannot be used.'], ...
        nProbe, ME.message);
end
if ~isequal(size(A), [nProbe, cfg.dims.nAction])
    error('makePolicyFcn:batchShape', ...
        ['Batched policy returned %s; expected [%d %d]. The approximator ', ...
         'output was probably left as an N-D array -- collapse the trailing ', ...
         'dimensions into the batch dimension before transposing.'], ...
        mat2str(size(A)), nProbe, cfg.dims.nAction);
end
if ~all(isfinite(A(:)))
    error('makePolicyFcn:nonFinite', 'Batched policy returned non-finite actions.');
end
end

% =====================================================================
function A = slowGetAction(agent, X, cfg)
%SLOWGETACTION  Last-resort per-row evaluation.
n = size(X,1);
A = zeros(n, cfg.dims.nAction);
for i = 1:n
    a = getAction(agent, X(i,:)');
    if iscell(a), a = a{1}; end
    A(i,:) = double(a(:))';
end
A = max(-1, min(1, A));
end

% =====================================================================
function v = extractdata_safe(v)
%EXTRACTDATA_SAFE  Unwrap a dlarray if Deep Learning Toolbox produced one.
if isa(v, 'dlarray')
    v = extractdata(v);
end
end

% =====================================================================
function A = validateOut(A, n, cfg)
%VALIDATEOUT  Fail loudly on the wrong shape rather than silently later.
if size(A,1) ~= n || size(A,2) ~= cfg.dims.nAction
    error('makePolicyFcn:badShape', ...
        ['Policy returned a %dx%d matrix; expected %dx%d. The policy must ', ...
         'map [n x nObs] to [n x nAction].'], ...
        size(A,1), size(A,2), n, cfg.dims.nAction);
end
end
