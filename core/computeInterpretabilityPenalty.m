function [penalty, breakdown] = computeInterpretabilityPenalty(state, dev, devPrev, feeder, cfg)
%COMPUTEINTERPRETABILITYPENALTY  Core innovation #1.
%
%   [P, BREAKDOWN] = COMPUTEINTERPRETABILITYPENALTY(STATE, DEV, DEVPREV,
%   FEEDER, CFG) returns a scalar penalty that is *large* when the action
%   just taken would be hard for a human dispatcher to justify, and near
%   zero when the action has an obvious local physical rationale.
%
%   Adding this term to the reward steers PPO toward intrinsically
%   explainable behaviour, so the post-hoc SHAP + LLM layer has an easier
%   policy to describe.
%
%   DESIGN CONSTRAINT: THIS FUNCTION MUST NOT CALL SHAP
%   ---------------------------------------------------
%   SHAP explains the policy, and the policy is being shaped by the penalty -- and it is
%   computationally impossible: the reward is evaluated once per
%   environment step (order 1e5-1e6 times during training) while a single
%   SHAP explanation costs thousands of policy forward passes. The penalty
%   below instead uses four physically grounded, O(nDevices^2) quantities
%   that correlate with what makes an action explainable.
%
%   THE FOUR TERMS
%   --------------
%     p1 CARDINALITY  -- number of devices moved beyond a free allowance.
%        An operator can follow "one bank closed"; they cannot follow
%        "seven devices moved simultaneously".
%
%     p2 SPATIAL COHERENCE -- for every pair of devices that moved in the
%        same step, a cost proportional to the hop distance between them,
%        waived when both of their local zones actually had a voltage
%        problem. Co-movement of electrically distant devices with only
%        one driver is the classic unexplainable action.
%
%     p3 PHYSICAL CONSISTENCY -- a move is penalized when its reactive
%        injection pushes local voltage in the wrong direction, i.e.
%        injecting VArs where voltage is already high, or absorbing them
%        where voltage is already low. Uses the sign of the local voltage
%        error, so it is exactly the sentence a dispatcher would object to.
%
%     p4 HYSTERESIS -- reversing the direction of a device relative to the
%        previous step. Chattering is both physically damaging and
%        narratively incoherent ("why did it close then open?").
%
%   Inputs
%   ------
%     STATE   : struct with field Vm ([nBus x 1] voltage magnitudes BEFORE
%               the action is applied -- the information the agent acted on)
%     DEV     : decoded device struct for the current action (DECODEACTION)
%     DEVPREV : decoded device struct for the previous action ([] at t=0)
%     FEEDER  : from BUILDFEEDER (provides hopDist and zone)
%     CFG     : from XRLCONFIG
%
%   Outputs
%   -------
%     PENALTY   : scalar, >= 0
%     BREAKDOWN : struct with fields p1..p4, moved, movedBuses -- retained
%                 because the LLM prompt quotes it when explaining *why*
%                 the agent avoided a more aggressive action.
%
%   See also COMPUTEREWARD, DECODEACTION.

Vm   = state.Vm(:);
nCap = numel(dev.capOn);
nPV  = numel(dev.invQ);

% ---------------------------------------------------------------------
% Assemble a common device list: bus, reactive delta, and "did it move".
% Devices are compared against the previous step; at t = 0 nothing has
% moved by definition except relative to the all-off reference.
% ---------------------------------------------------------------------
if isempty(devPrev)
    devPrev.tap   = 0;
    devPrev.capQ  = zeros(nCap,1);
    devPrev.invQ  = zeros(nPV,1);
    devPrev.capOn = zeros(nCap,1);
end

busList  = [cfg.oltc.bus; dev.capBuses(:); dev.pvBuses(:)];

% Signed "reactive-equivalent" move of each device. For the OLTC we use
% the tap change directly (a tap raise acts like a VAr injection at the
% head of the feeder, so the sign convention matches).
dQ = [ (dev.tap - devPrev.tap) * cfg.oltc.stepPU; ...
        dev.capQ(:) - devPrev.capQ(:); ...
        dev.invQ(:) - devPrev.invQ(:) ];

nDev = numel(dQ);

% A device "moved" if its change exceeds a small deadband. The deadband
% stops floating-point dust in the continuous inverter channels from
% counting as an operator-visible action.
deadband = [0.5*cfg.oltc.stepPU; ...
            0.5*min(cfg.cap.kvar(:))/1000/cfg.net.baseMVA * ones(nCap,1); ...
            0.02 * ones(nPV,1)];
moved = abs(dQ) > deadband;

% ---------------------------------------------------------------------
% p1 -- cardinality beyond the free allowance
% ---------------------------------------------------------------------
nMoved = sum(moved);
p1 = max(0, nMoved - cfg.interp.maxSimultaneous);

% ---------------------------------------------------------------------
% p2 -- spatial coherence of co-moving devices
% ---------------------------------------------------------------------
p2 = 0;
idxMoved = find(moved);
if numel(idxMoved) >= 2
    for ii = 1:numel(idxMoved)-1
        for jj = ii+1:numel(idxMoved)
            bi = busList(idxMoved(ii));
            bj = busList(idxMoved(jj));
            d  = feeder.hopDist(bi, bj);
            if d <= cfg.interp.hopThreshold
                continue    % electrically close: co-movement is natural
            end
            % Distant pair. Waive the cost only if BOTH ends genuinely
            % needed help -- then there are two independent drivers and
            % the action is still explainable as two local decisions.
            needI = localNeed(bi, Vm, feeder, cfg);
            needJ = localNeed(bj, Vm, feeder, cfg);
            if needI && needJ
                continue
            end
            p2 = p2 + (d - cfg.interp.hopThreshold) / feeder.diameter;
        end
    end
end

% ---------------------------------------------------------------------
% p3 -- physical consistency of each move with its local voltage error
% ---------------------------------------------------------------------
% Local voltage error, normalised by the soft band half-width. Positive
% means "locally too high", negative means "locally too low".
halfBand = (cfg.limits.VsoftMax - cfg.limits.VsoftMin) / 2;
p3 = 0;
for k = idxMoved(:)'
    b     = busList(k);
    vLoc  = localVoltage(b, Vm, feeder);
    vErr  = (vLoc - cfg.limits.Vnom) / halfBand;
    % A move is inconsistent when dQ and vErr share a sign: raising VArs
    % (dQ > 0) while the local voltage is already high (vErr > 0), or
    % absorbing VArs while it is already low.
    incons = max(0, sign(dQ(k)) * vErr);
    % Weight by the magnitude of the move, normalised so a full-range move
    % of any device type contributes comparably.
    scaleK = max(deadband(k)*4, eps);
    p3 = p3 + incons * min(1, abs(dQ(k)) / scaleK);
end

% ---------------------------------------------------------------------
% p4 -- direction reversal versus the previous step (chattering)
% ---------------------------------------------------------------------
p4 = 0;
if isfield(devPrev, 'dQ') && ~isempty(devPrev.dQ)
    prevDir = sign(devPrev.dQ(:));
    curDir  = sign(dQ);
    reversals = (prevDir ~= 0) & (curDir ~= 0) & (prevDir ~= curDir);
    p4 = sum(reversals);
end

% ---------------------------------------------------------------------
% Weighted sum
% ---------------------------------------------------------------------
penalty = cfg.interp.wCardinality * p1 + ...
          cfg.interp.wCoherence   * p2 + ...
          cfg.interp.wPhysical    * p3 + ...
          cfg.interp.wHysteresis  * p4;

breakdown.p1         = p1;
breakdown.p2         = p2;
breakdown.p3         = p3;
breakdown.p4         = p4;
breakdown.nMoved     = nMoved;
breakdown.moved      = moved;
breakdown.movedBuses = busList(moved);
breakdown.dQ         = dQ;
breakdown.total      = penalty;
end

% =====================================================================
function v = localVoltage(bus, Vm, feeder)
%LOCALVOLTAGE  Mean voltage in the electrical neighbourhood of a bus.
%   Using a neighbourhood rather than the single bus makes the measure
%   robust to one noisy meter and matches how an operator reads a feeder.
neigh = feeder.hopDist(bus,:) <= 2;
v = mean(Vm(neigh));
end

% =====================================================================
function tf = localNeed(bus, Vm, feeder, cfg)
%LOCALNEED  True when the neighbourhood of BUS is outside the soft band.
v = localVoltage(bus, Vm, feeder);
tf = (v < cfg.limits.VsoftMin) || (v > cfg.limits.VsoftMax);
end
