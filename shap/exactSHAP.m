function [phi, out] = exactSHAP(predictFcn, X, background, groups, opts)
%EXACTSHAP  Exact Shapley values by full coalition enumeration.
%
%   [PHI, OUT] = EXACTSHAP(PREDICTFCN, X, BACKGROUND, GROUPS, OPTS)
%
%   Computes exact Shapley values over M = numel(GROUPS) players by
%   enumerating all 2^M coalitions. This is the grouped adaptation of the
%   exact-SHAP routine supplied in the project brief, with three changes
%   that were necessary to make it usable here:
%
%     1. PLAYERS ARE GROUPS, NOT RAW FEATURES. The brief's snippet
%        enumerates 2^M coalitions over raw features. With M = 80
%        observation channels that is 2^80 coalitions -- not slow, but
%        physically impossible. Grouping the 80 features into 12 semantic
%        blocks (FEATUREMAPPING) brings it to 4096, which enumerates in
%        seconds and produces attributions an operator can actually read.
%
%     2. BATCHED EVALUATION. The snippet loops over background samples one
%        at a time inside a loop over coalitions, i.e. 2^M * nBG separate
%        model calls. Here COALITIONVALUES builds the hybrid samples as
%        matrices and calls the policy on chunks of ~20k rows, which is
%        two to three orders of magnitude faster in MATLAB.
%
%     3. VECTORIZED MARGINAL CONTRIBUTIONS. The pairing of each coalition
%        S with S u {j} is done with bit arithmetic over the whole
%        coalition index vector at once, as in the brief, but with the bit
%        order corrected so that column j of Z corresponds to bit j.
%
%   The Shapley weight uses the GAMMALN form from the brief,
%       w(k) = k! (M-k-1)! / M!  =  exp(gammaln(k+1)+gammaln(M-k)-gammaln(M+1)),
%   which stays finite for M well beyond anything enumerable.
%
%   Inputs
%   ------
%     PREDICTFCN : handle, [n x nFeat] -> [n x 1]
%     X          : [nExplain x nFeat] instances to explain
%     BACKGROUND : [nBG x nFeat] background distribution
%     GROUPS     : {1 x M} cell of feature-index vectors
%     OPTS       : struct, fields maxExactM (default 16), batchSize (20000),
%                  verbose (false)
%
%   Outputs
%   -------
%     PHI : [nExplain x M] exact Shapley values per instance and group
%     OUT : struct with .phi0 (background mean), .fx (instance values),
%           .efficiencyGap (should be ~1e-12; see below), .M, .nCoalitions
%
%   THE EFFICIENCY CHECK IS NOT OPTIONAL
%   ------------------------------------
%   Exact Shapley values satisfy sum_j phi_j = f(x) - E[f(bg)] identically.
%   OUT.efficiencyGap reports the residual. If it is not at machine
%   precision, something is wrong -- almost always a PREDICTFCN that is
%   stochastic (a sampling policy rather than the deterministic mean) or
%   not row-wise. Check it before trusting a single number downstream.
%
%   See also KERNELSHAP, COALITIONVALUES, FEATUREMAPPING.

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'maxExactM'), opts.maxExactM = 16;    end
if ~isfield(opts,'batchSize'), opts.batchSize = 20000; end
if ~isfield(opts,'verbose'),   opts.verbose   = false; end

M = numel(groups);
if M > opts.maxExactM
    error('exactSHAP:tooManyGroups', ...
        ['Exact enumeration needs 2^%d = %.3g coalitions, above the limit ', ...
         'maxExactM = %d. Either coarsen the grouping or switch to ', ...
         'KERNELSHAP (cfg.shap.mode = ''raw'').'], M, 2^M, opts.maxExactM);
end

nExplain = size(X,1);
nCoal    = 2^M;

% ---------------------------------------------------------------------
% Coalition matrix. Column j holds bit (j-1) of the coalition index, so
% coalition number n and column j are related by bitget(n, j). Keeping
% that convention explicit avoids the classic dec2bin ordering bug.
% ---------------------------------------------------------------------
coalNum = (0:nCoal-1)';
Z = false(nCoal, M);
for j = 1:M
    Z(:,j) = bitget(coalNum, j) > 0;
end
k_vals = sum(Z, 2);

% ---------------------------------------------------------------------
% Stable Shapley weights w(k) = k!(M-k-1)!/M!
% ---------------------------------------------------------------------
weight = zeros(nCoal,1);
valid  = k_vals < M;
weight(valid) = exp( gammaln(k_vals(valid)+1) + gammaln(M-k_vals(valid)) - gammaln(M+1) );

phi = zeros(nExplain, M);
fx  = zeros(nExplain, 1);
phi0 = NaN;

for i = 1:nExplain
    if opts.verbose
        fprintf('  exactSHAP: instance %d/%d\n', i, nExplain);
    end

    fAll = coalitionValues(predictFcn, X(i,:), background, Z, groups, opts.batchSize);

    % v(empty) = E_bg[f(bg)]; v(full) = f(x)
    if i == 1
        phi0 = fAll(1);
    end
    fx(i) = fAll(end);

    % ---- Marginal contributions, vectorised over coalitions ----------
    for j = 1:M
        notJ    = ~Z(:,j);
        numNot  = coalNum(notJ);
        numWith = bitset(numNot, j);          % add player j to the coalition
        phi(i,j) = sum( weight(numNot+1) .* ( fAll(numWith+1) - fAll(numNot+1) ) );
    end
end

out.phi0          = phi0;
out.fx            = fx;
out.M             = M;
out.nCoalitions   = nCoal;
out.efficiencyGap = max(abs(sum(phi,2) - (fx - phi0)));
out.method        = 'exact-grouped-interventional';

% ATTRIBUTION NOISE FLOOR.
% The efficiency residual is a direct measure of the numerical error that
% accumulated through the whole computation, so it doubles as the smallest
% attribution that can be distinguished from rounding. This matters
% because MATLAB's dlnetwork evaluates in SINGLE precision by default
% (eps ~ 1.2e-7): a policy whose outputs carry ~1e-7 of noise cannot
% support attributions below roughly that scale, however many decimal
% places the number is printed with.
%
% On this project the floor lands near 1e-7 while real attributions are
% 1e-2 to 3e-2, so the margin is ~5 orders of magnitude -- the check is
% cheap insurance, not a live constraint.
out.noiseFloor = max(out.efficiencyGap, eps('single'));
out.maxAbsPhi  = max(abs(phi(:)));
out.snr        = out.maxAbsPhi / max(out.noiseFloor, realmin);

if out.efficiencyGap > 1e-6
    warning('exactSHAP:efficiency', ...
        ['Efficiency residual %.3e exceeds tolerance. The usual cause is a ', ...
         'stochastic predictFcn -- explain the policy MEAN, not a sample ', ...
         '(see SHAPVALUEFUNCTION).'], out.efficiencyGap);
end
end
