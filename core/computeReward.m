function [reward, info] = computeReward(state, pf, dev, devPrev, feeder, cfg)
%COMPUTEREWARD  Multi-objective VVC reward with interpretability penalty.
%
%   R = -( alpha*Vviol + beta*Ploss + gamma*ActionChange + delta*Interp )
%
%   Inputs
%   ------
%     STATE   : pre-action state struct (field Vm used by the penalty)
%     PF      : post-action power-flow solution from RADIALPOWERFLOW
%     DEV     : decoded device struct for this step
%     DEVPREV : decoded device struct for the previous step ([] at t = 0)
%     FEEDER  : from BUILDFEEDER
%     CFG     : from XRLCONFIG
%
%   Outputs
%   -------
%     REWARD : scalar
%     INFO   : struct with every term separated out, so training curves can
%              be decomposed and the ablation study can switch terms off
%              without touching this file.
%
%   TERM DEFINITIONS
%   ----------------
%   Vviol  : sum of squared excursions beyond [Vmin, Vmax], summed over
%            buses. Squared (not absolute) so that a single deep violation
%            is punished far harder than many marginal ones -- this is what
%            an operator cares about.
%   Ploss  : total active loss in p.u. on the 10 MVA base.
%   Action : L1 change of the normalized action vector, a proxy for device
%            wear that treats a tap move and a bank switch on a common scale.
%   Interp : COMPUTEINTERPRETABILITYPENALTY.
%
%   See also COMPUTEINTERPRETABILITYPENALTY, XRLSTEP.

Vm = pf.Vm(:);

% ---- Voltage violation ----------------------------------------------
under = max(0, cfg.limits.Vmin - Vm);
over  = max(0, Vm - cfg.limits.Vmax);
Vviol = sum(under.^2) + sum(over.^2);

% ---- Active power loss ----------------------------------------------
Ploss = pf.Ploss;

% ---- Device wear -----------------------------------------------------
if isempty(devPrev)
    actionChange = 0;
else
    actionChange = sum(abs(dev.aClipped - devPrev.aClipped));
end

% ---- Interpretability penalty (core innovation #1) -------------------
% ALWAYS compute the breakdown, even when its weight is zero.
%
% The previous version skipped the computation entirely when the weight
% was zero and substituted a stub with nMoved = 0. That stub is a
% DIAGNOSTIC, not just a reward term: COLLECTATTRIBUTIONS selects which
% steps to explain via interpBreakdown.nMoved > 0, and the dashboard and
% the LLM prompt both read the breakdown. Zeroing the weight therefore
% silently disabled SHAP collection -- visible in the sweep as
% "the policy never moved a device" alongside a switching rate of 0.256.
%
% Computing it unconditionally costs O(nDevices^2) on nine devices, which
% is negligible, and keeps the diagnostics independent of the weighting.
[interp, interpBreakdown] = computeInterpretabilityPenalty( ...
    state, dev, devPrev, feeder, cfg);

% ---- Weighted sum ----------------------------------------------------
% SCALE-FREE FORM (default). Each term is divided by a fixed physical
% REFERENCE before weighting, so the weights are pure relative priorities
% and are independent of units and of how well the policy happens to be
% doing.
%
% WHY THIS REPLACED RAW WEIGHTS
% -----------------------------
% Two full training runs were lost to raw-weight miscalibration, in
% opposite directions, and neither was visible in the training curve:
%
%   alpha=10, delta=0.05   -> interpretability was 91% of the signal. The
%                             agent froze the capacitor banks and accepted
%                             52x the rule baseline's violations.
%   alpha=200, delta=0.02  -> interpretability still 76%, loss only 12%.
%                             The agent drove all three inverters to near
%                             maximum VAr injection, achieving ZERO
%                             violations at 177 kW of loss against the rule
%                             baseline's 60 kW -- a 193% regression, worth
%                             about 1 GWh/year.
%
% The common cause is that a raw weight only means something relative to
% the magnitude the term actually takes, and those magnitudes move by
% orders of magnitude as the policy learns (Vviol collapses toward zero;
% Ploss and the penalties do not). Normalizing by a fixed reference makes
% cfg.reward.w read as "voltage matters 3x more than loss", which is a
% statement about engineering priorities that survives retuning.
if isfield(cfg.reward,'normalize') && cfg.reward.normalize
    R = cfg.reward.ref;
    W = cfg.reward.w;
    termV = W.voltage * Vviol        / R.Vviol;
    termP = W.loss    * Ploss        / R.Ploss;
    termA = W.wear    * actionChange / R.action;
    termI = W.interp  * interp       / R.interp;
    reward = -cfg.reward.scale * (termV + termP + termA + termI);
else
    % Legacy raw-weight form, retained so agents trained before the change
    % can still be evaluated against the reward they were trained on.
    termV = cfg.reward.alpha * Vviol;
    termP = cfg.reward.beta  * Ploss;
    termA = cfg.reward.gamma * actionChange;
    termI = cfg.reward.delta * interp;
    reward = -cfg.reward.scale * (termV + termP + termA + termI);
end

% Hard safety backstop: a non-converged power flow means the commanded
% operating point is not physical. Return a large negative reward rather
% than propagating NaNs into PPO's advantage estimates.
if ~pf.converged || any(~isfinite(Vm))
    reward = -100;
end

info.Vviol          = Vviol;
info.Ploss          = Ploss;
info.PlossKW        = Ploss * cfg.net.baseMVA * 1000;
info.actionChange   = actionChange;
info.interp         = interp;
info.interpBreakdown= interpBreakdown;
info.nViolatedBuses = nnz(under > 0 | over > 0);
info.Vmin           = min(Vm);
info.Vmax           = max(Vm);
info.reward         = reward;
info.termVoltage    = termV;
info.termLoss       = termP;
info.termWear       = termA;
info.termInterp     = termI;
end
