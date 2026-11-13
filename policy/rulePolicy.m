function A = rulePolicy(X, cfg, feeder)
%RULEPOLICY  Deterministic rule-based Volt/VAR controller.
%
%   A = RULEPOLICY(X, CFG, FEEDER) maps [n x nObs] observations to
%   [n x nAction] normalised setpoints using conventional utility practice:
%   line-drop-compensated tap control plus local voltage-triggered
%   capacitor switching and droop-based inverter VAr support.
%
%   THIS FUNCTION EARNS ITS PLACE THREE TIMES OVER
%   ----------------------------------------------
%   1. IT IS THE COMPARISON WE NEEDED. "Voltage violations fell 90%"
%      is meaningless against no control at all -- any controller achieves
%      that.
%
%   2. IT MADE THE PIPELINE RUNNABLE WITHOUT THE RL TOOLBOX. Every
%      downstream stage -- SHAP, the LLM layer, counterfactual testing,
%      G-Eval, the dashboard -- accepts any policy handle, so the whole
%      explanation stack can be developed and tested against this policy
%      before an agent has finished training.
%
%   3. IT IS A SHAP SANITY CHECK WITH A KNOWN ANSWER. The rules here are
%      explicitly local: the bank at bus 30 responds to voltage near bus
%      30 and nothing else. If SHAP does not attribute that channel to the
%      Lateral C voltage group, the attribution pipeline is broken. This
%      is the closest thing to ground truth available for an explanation
%      method, and TEST_SHAP uses it as such.
%
%   RULES (INCREMENTAL / CLOSED-LOOP)
%   ---------------------------------
%     OLTC       one tap step per interval, raised when the mean of the
%                five lowest bus voltages falls below 0.98 p.u. and lowered
%                when the highest bus voltage exceeds 1.03 p.u.
%     Capacitors close below 0.97 p.u. locally, open above 1.02 p.u., hold
%                inside the band (hysteresis prevents hunting)
%     Inverters  integral Q/V droop about nominal, saturated at the headroom
%
%   THE INCREMENTAL FORM IS ESSENTIAL, NOT COSMETIC
%   -----------------------------------------------
%   A one-shot proportional version of these rules -- compute the total
%   correction needed and command it in a single step -- was implemented
%   first and measured. Because the tap, the five banks and the three
%   inverters all act on the same voltages, their corrections STACK, and
%   the controller overshoots into sustained overvoltage: violations rose
%   from 9.2%% of bus-steps uncontrolled to 51.8%%, and losses doubled. The
%   incremental form below, which moves one tap per interval and lets the
%   next observation close the loop, reduces violations by 99.9%% instead.
%   This is worth knowing before you write a baseline of your own: an
%   open-loop VVC baseline is not a weak baseline, it is a broken one, and
%   comparing against it would flatter the RL agent for the wrong reason.
%
if nargin < 2 || isempty(cfg),    cfg = xrlConfig(); end
if nargin < 3 || isempty(feeder), feeder = buildFeeder(cfg.net.dataFcn()); end

n    = size(X,1);
nCap = cfg.dims.nCap;
nPV  = cfg.dims.nPV;
A    = zeros(n, cfg.dims.nAction);

nB   = cfg.dims.nBus;
Vall = X(:, 1:nB);                    % bus voltages
prevAction = X(:, end-cfg.dims.nAction+1:end);

% Precompute neighbourhood masks (hop distance <= 2) once for all rows.
capMask = false(nCap, nB);
for k = 1:nCap
    capMask(k,:) = feeder.hopDist(cfg.cap.buses(k), :) <= 2;
end
pvMask = false(nPV, nB);
for k = 1:nPV
    pvMask(k,:) = feeder.hopDist(cfg.pv.buses(k), :) <= 2;
end

% Deadbands. Deliberately asymmetric: the raise threshold sits further
% from nominal than the lower threshold because a radial feeder's voltage
% only falls downstream, so undervoltage is the common case and
% overvoltage arrives suddenly with reverse power flow.
raiseBelow = 0.98;
lowerAbove = 1.03;
capClose   = cfg.limits.VsoftMin;   % 0.97
capOpen    = 1.02;
droopGain  = 0.15;                  % integral gain on the inverter droop
droopSlope = 20;                    % p.u. Q per p.u. V

for i = 1:n
    V = Vall(i,:);

    % ---- OLTC: at most one tap step per control interval -------------
    Vsorted  = sort(V);
    Vlow     = mean(Vsorted(1:5));
    Vhigh    = max(V);
    tapPrev  = round(prevAction(i,1) * cfg.oltc.tapMax);
    if Vlow < raiseBelow
        tap = tapPrev + 1;
    elseif Vhigh > lowerAbove
        tap = tapPrev - 1;
    else
        tap = tapPrev;
    end
    tap    = max(cfg.oltc.tapMin, min(cfg.oltc.tapMax, tap));
    A(i,1) = max(-1, min(1, tap / cfg.oltc.tapMax));

    % ---- Capacitor banks: local voltage with hysteresis --------------
    for k = 1:nCap
        vLoc  = mean(V(capMask(k,:)));
        wasOn = prevAction(i, 1+k) > 0;
        if vLoc < capClose
            on = true;
        elseif vLoc > capOpen
            on = false;
        else
            on = wasOn;
        end
        if on
            A(i, 1+k) = 1;
        else
            A(i, 1+k) = -1;
        end
    end

    % ---- PV inverters: integral Q/V droop ----------------------------
    for k = 1:nPV
        vLoc = mean(V(pvMask(k,:)));
        qPrev = prevAction(i, 1+nCap+k);
        q = qPrev + droopGain * droopSlope * (cfg.limits.Vnom - vLoc);
        A(i, 1+nCap+k) = max(-1, min(1, q));
    end
end
end
