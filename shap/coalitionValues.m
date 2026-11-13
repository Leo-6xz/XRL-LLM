function fz = coalitionValues(predictFcn, x, background, Z, groups, batchSize)
%COALITIONVALUES  Interventional value function shared by both SHAP engines.
%
%   FZ = COALITIONVALUES(PREDICTFCN, X, BACKGROUND, Z, GROUPS, BATCHSIZE)
%   evaluates, for every coalition (row of Z),
%
%       v(S) = E_bg [ f( x_S , bg_{-S} ) ]
%
%   i.e. features in the coalition keep their value from the instance X,
%   and features outside it are REPLACED by the corresponding value from
%   each background sample, with the result averaged over the background.
%
%   Inputs
%   ------
%     PREDICTFCN : handle, [n x nFeat] -> [n x 1] scalar output
%     X          : [1 x nFeat] the instance being explained
%     BACKGROUND : [nBG x nFeat] background distribution
%     Z          : [nCoal x M] binary coalition matrix over GROUPS
%     GROUPS     : {1 x M} cell, each holding feature indices in that group
%     BATCHSIZE  : max rows per PREDICTFCN call
%
%   Output
%   ------
%     FZ : [nCoal x 1] coalition values
%
%   INTERVENTIONAL, NOT CONDITIONAL
%   -------------------------------
%   Replacing out-of-coalition features with independent background draws
%   implements the *interventional* (a.k.a. marginal) Shapley value. The
%   alternative, conditional SHAP, would require modelling p(x_{-S} | x_S),
%   which on a distribution feeder means modelling the power flow itself.
%   Interventional values answer "what does the policy do if I intervene
%   on these inputs", which is the question an operator is actually asking
%   and the one COUNTERFACTUALTEST can verify.
%
%   See also EXACTSHAP, KERNELSHAP.

if nargin < 6 || isempty(batchSize)
    batchSize = 20000;
end

x   = x(:)';
nBG = size(background, 1);
[nCoal, M] = size(Z);

if numel(groups) ~= M
    error('coalitionValues:groupMismatch', ...
        'Z has %d columns but %d groups were supplied.', M, numel(groups));
end

% Precompute a [M x nFeat] logical membership matrix so building each
% coalition mask is one matrix multiply instead of a loop over groups.
nFeat = numel(x);
memberMask = false(M, nFeat);
for g = 1:M
    memberMask(g, groups{g}) = true;
end

fz = zeros(nCoal, 1);

% Process coalitions in blocks such that block * nBG <= batchSize.
coalPerBlock = max(1, floor(batchSize / nBG));

for c0 = 1:coalPerBlock:nCoal
    c1  = min(nCoal, c0 + coalPerBlock - 1);
    idx = c0:c1;
    nC  = numel(idx);

    % Feature-level keep mask for each coalition in the block: [nC x nFeat]
    keep = Z(idx,:) * memberMask > 0;

    % Build [nC*nBG x nFeat] hybrid samples. Row ordering is
    % (coalition-major, background-minor).
    keepRep = repelem(keep, nBG, 1);                 % [nC*nBG x nFeat]
    bgRep   = repmat(background, nC, 1);             % [nC*nBG x nFeat]
    xRep    = repmat(x, nC*nBG, 1);

    Xhyb = bgRep;
    Xhyb(keepRep) = xRep(keepRep);

    y = predictFcn(Xhyb);
    y = y(:);
    if numel(y) ~= nC*nBG
        error('coalitionValues:badPredict', ...
            ['predictFcn must return one scalar per input row: expected %d, ', ...
             'got %d. Wrap your policy with SHAPVALUEFUNCTION.'], nC*nBG, numel(y));
    end

    % Average over the background for each coalition.
    fz(idx) = mean(reshape(y, nBG, nC), 1)';
end
end
