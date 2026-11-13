function [predictFcn, meta] = shapValueFunction(policyFcn, actionIdx, cfg)
%SHAPVALUEFUNCTION  Wrap a vector-output policy into a scalar SHAP target.
%
%   [PREDICTFCN, META] = SHAPVALUEFUNCTION(POLICYFCN, ACTIONIDX, CFG)
%
%   SHAP requires a scalar-valued function. A VVC policy emits 9 numbers.
%   This wrapper selects what is being explained:
%
%     ACTIONIDX = integer k  -> explain action channel k (e.g. "why did the
%                               bank at bus 30 close?"). This is the mode
%                               used for per-decision operator explanations.
%     ACTIONIDX = 'norm'     -> explain the L2 magnitude of the action, i.e.
%                               "why did the agent intervene at all?"
%     ACTIONIDX = 'oltc'     -> shorthand for channel 1.
%
%   TWO REQUIREMENTS ON POLICYFCN
%   -----------------------------
%   1. ROW-WISE BATCH: it must accept [n x nObs] and return [n x nAction].
%      COALITIONVALUES calls it with ~20000 rows at a time; a policy that
%      only handles one observation makes SHAP roughly 1000x slower.
%   2. DETERMINISTIC: it must return the policy MEAN, not a sample from the
%      Gaussian. Explaining a stochastic draw makes the Shapley efficiency
%      axiom fail and the attributions become noise. MAKEPOLICYFCN handles
%      this by calling EVALUATE on the actor rather than GETACTION.
%
%   See also MAKEPOLICYFCN, EXACTSHAP, KERNELSHAP.

if nargin < 3 || isempty(cfg), cfg = xrlConfig(); end

meta.actionIdx = actionIdx;

if ischar(actionIdx) || isstring(actionIdx)
    switch lower(char(actionIdx))
        case 'norm'
            predictFcn = @(X) sqrt(sum(policyFcn(X).^2, 2));
            meta.name  = 'Overall control effort (L2 norm of action)';
            meta.units = '-';
            return
        case 'oltc'
            actionIdx = 1;
        otherwise
            error('shapValueFunction:badTarget', ...
                'ACTIONIDX must be an integer, ''norm'' or ''oltc''.');
    end
end

validateattributes(actionIdx, {'numeric'}, ...
    {'scalar','integer','>=',1,'<=',cfg.dims.nAction}, mfilename, 'actionIdx');

predictFcn = @(X) indexColumn(policyFcn(X), actionIdx);

names      = featureMapping(cfg);
meta.name  = names.actionNames{actionIdx};
meta.units = 'normalised setpoint';
meta.actionIdx = actionIdx;
end

function y = indexColumn(A, k)
if size(A,2) < k
    error('shapValueFunction:policyShape', ...
        ['policyFcn returned %d columns; expected at least %d. It must map ', ...
         '[n x nObs] -> [n x nAction].'], size(A,2), k);
end
y = A(:,k);
end
