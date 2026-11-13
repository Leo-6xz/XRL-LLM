function [phi, out] = kernelSHAP(predictFcn, X, background, groups, opts)
%KERNELSHAP  Model-agnostic Shapley attribution (Lundberg & Lee, 2017).
%
%   [PHI, OUT] = KERNELSHAP(PREDICTFCN, X, BACKGROUND, GROUPS, OPTS)
%
%   Core innovation #2 of the project. KernelSHAP is used rather than
%   TreeSHAP because the object being explained is a deep PPO policy
%   network, for which no tree structure exists to exploit.
%
%   THREE CORRECTIONS RELATIVE TO THE DRAFT IN THE PROJECT BRIEF
%   -----------------------------------------------------------
%   The brief's kernelSHAP sketch solves
%       phi = (Z' W Z) \ (Z' W f)
%   which is wrong in three ways, each of which silently produces
%   plausible-looking but meaningless numbers:
%
%     (a) NO INTERCEPT. The regression must explain f(z) - phi0 where
%         phi0 = E_bg[f]. Regressing on the raw f absorbs the baseline
%         into the attributions and inflates every value.
%
%     (b) NO EFFICIENCY CONSTRAINT. Shapley values must satisfy
%         sum_j phi_j = f(x) - phi0. KernelSHAP enforces this by
%         constrained least squares. Without it the "large weight 1e6 on
%         the full and empty coalitions" trick in the sketch is a crude
%         penalty substitute that leaves the constraint violated by an
%         uncontrolled amount.
%
%     (c) SINGULAR DESIGN MATRIX. With the full and empty coalitions
%         included as rows, Z has a constant column combination and
%         Z'WZ is rank-deficient, so the backslash returns a
%         minimum-norm solution that depends on floating-point noise.
%
%   This implementation instead excludes the empty and full coalitions
%   from the regression (they are exactly the two constraints), and
%   imposes efficiency by eliminating the last player:
%
%       y   = v(z) - phi0 - z_M * (f(x) - phi0)
%       Zr  = Z(:,1:M-1) - Z(:,M)
%       phi(1:M-1) = argmin (Zr*b - y)' W (Zr*b - y)
%       phi(M)     = (f(x) - phi0) - sum(phi(1:M-1))
%
%   which is the standard closed form and satisfies both axioms exactly.
%
%   Inputs
%   ------
%     PREDICTFCN : handle, [n x nFeat] -> [n x 1]
%     X          : [nExplain x nFeat]
%     BACKGROUND : [nBG x nFeat]
%     GROUPS     : {1 x M} cell of feature indices; pass {} for one group
%                  per feature (the raw 80-feature attribution)
%     OPTS       : .nCoalitions (2048), .batchSize (20000), .ridge (1e-8),
%                  .verbose (false), .seed ([])
%
%   Outputs
%   -------
%     PHI : [nExplain x M]
%     OUT : .phi0, .fx, .efficiencyGap, .nCoalitions, .method
%
%   See also EXACTSHAP, COALITIONVALUES, SHAPVALUEFUNCTION.

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nCoalitions'), opts.nCoalitions = 2048;  end
if ~isfield(opts,'batchSize'),   opts.batchSize   = 20000; end
if ~isfield(opts,'ridge'),       opts.ridge       = 1e-8;  end
if ~isfield(opts,'verbose'),     opts.verbose     = false; end
if ~isfield(opts,'seed'),        opts.seed        = [];    end

nFeat = size(X,2);
if isempty(groups)
    groups = num2cell(1:nFeat);
end
M = numel(groups);

if M < 2
    error('kernelSHAP:tooFewPlayers', 'Need at least 2 players.');
end

if ~isempty(opts.seed)
    rs = RandStream('mt19937ar','Seed',opts.seed);
else
    rs = RandStream.getGlobalStream();
end

% ---------------------------------------------------------------------
% Build the coalition set (excluding empty and full -- those two are the
% constraints, not observations).
% ---------------------------------------------------------------------
maxCoal = 2^M - 2;
if maxCoal <= opts.nCoalitions
    % Small enough to enumerate: KernelSHAP then reproduces exact Shapley
    % values to machine precision, which TEST_SHAP relies on.
    coalNum = (1:2^M-2)';
    Z = false(numel(coalNum), M);
    for j = 1:M
        Z(:,j) = bitget(coalNum, j) > 0;
    end
    enumerated = true;
else
    Z = sampleCoalitions(M, opts.nCoalitions, rs);
    enumerated = false;
end

kz = sum(Z,2);

% ---------------------------------------------------------------------
% Shapley kernel weights: pi(z) = (M-1) / ( C(M,k) * k * (M-k) )
% Computed in log space; nchoosek overflows for M around 80.
% ---------------------------------------------------------------------
logC = gammaln(M+1) - gammaln(kz+1) - gammaln(M-kz+1);
logW = log(M-1) - logC - log(kz) - log(M-kz);
w    = exp(logW - max(logW));      % rescale: only relative weights matter
w    = w / sum(w);

nExplain = size(X,1);
phi = zeros(nExplain, M);
fx  = zeros(nExplain, 1);

% Baseline: v(empty) = E_bg[f(bg)]
phi0 = mean(predictFcn(background));

% Design matrix for the efficiency-eliminated regression
Zr = double(Z(:,1:M-1)) - double(Z(:,M));
W  = spdiags(w, 0, numel(w), numel(w));
A  = Zr' * W * Zr + opts.ridge * eye(M-1);

for i = 1:nExplain
    if opts.verbose
        fprintf('  kernelSHAP: instance %d/%d\n', i, nExplain);
    end

    fx(i) = predictFcn(X(i,:));
    vz    = coalitionValues(predictFcn, X(i,:), background, Z, groups, opts.batchSize);

    y = vz - phi0 - double(Z(:,M)) * (fx(i) - phi0);
    b = A \ (Zr' * W * y);

    phi(i,1:M-1) = b';
    phi(i,M)     = (fx(i) - phi0) - sum(b);
end

out.phi0          = phi0;
out.fx            = fx;
out.M             = M;
out.nCoalitions   = size(Z,1);
out.enumerated    = enumerated;
out.efficiencyGap = max(abs(sum(phi,2) - (fx - phi0)));
% See EXACTSHAP for why the efficiency residual is used as the attribution
% noise floor. KernelSHAP imposes efficiency as a constraint, so its
% residual is near zero by construction; fall back to the precision limit.
out.noiseFloor    = max(out.efficiencyGap, eps('single'));
out.maxAbsPhi     = max(abs(phi(:)));
out.snr           = out.maxAbsPhi / max(out.noiseFloor, realmin);
if enumerated
    out.method = 'kernelSHAP-enumerated-interventional';
else
    out.method = 'kernelSHAP-sampled-interventional';
end
end

% =====================================================================
function Z = sampleCoalitions(M, n, rs)
%SAMPLECOALITIONS  Draw coalitions with sizes distributed by the kernel.
%
%   The Shapley kernel puts almost all of its mass on very small and very
%   large coalitions. Sampling coalition SIZES from that distribution and
%   then choosing members uniformly at random within a size is far more
%   efficient than sampling the 2^M coalitions uniformly, which would put
%   nearly all draws at k ~ M/2 where the kernel weight is negligible.

k = (1:M-1)';
logC   = gammaln(M+1) - gammaln(k+1) - gammaln(M-k+1);
logPk  = log(M-1) - log(k) - log(M-k);   % kernel weight * C(M,k)
pk     = exp(logPk - max(logPk));
pk     = pk / sum(pk);
cdf    = cumsum(pk);

Z = false(n, M);
u = rand(rs, n, 1);
for i = 1:n
    ki = find(cdf >= u(i), 1, 'first');
    idx = randperm(rs, M, k(ki));
    Z(i, idx) = true;
end

% Guard: degenerate draws would break the regression.
bad = all(Z,2) | ~any(Z,2);
if any(bad)
    Z(bad,:) = false;
    for i = find(bad)'
        Z(i, randperm(rs, M, max(1, floor(M/2)))) = true;
    end
end
end
