function dev = decodeAction(a, cfg, pvPnow)
%DECODEACTION  Map a normalized PPO action vector to physical setpoints.
%
%   DEV = DECODEACTION(A, CFG, PVPNOW) converts the agent's raw output
%   A (a 9-element vector in [-1,1]) into device commands.
%
%   Inputs
%   ------
%     A      : [9 x 1] normalised action
%              a(1)     -> OLTC tap
%              a(2:6)   -> capacitor banks (5)
%              a(7:9)   -> PV inverter reactive setpoints (3)
%     CFG    : configuration struct from XRLCONFIG
%     PVPNOW : [3 x 1] current PV active output in p.u. Needed because the
%              inverter's reactive headroom shrinks as active output rises.
%
%   Output DEV
%   ----------
%     .tap      integer tap position in [tapMin, tapMax]
%     .V0       resulting slack voltage (p.u.)
%     .capOn    [nCap x 1] logical (or fractional if cap.binary is false)
%     .capQ     [nCap x 1] reactive injection per bank (p.u.)
%     .invQ     [nPV x 1] inverter reactive injection (p.u., + = injecting)
%     .invQmax  [nPV x 1] available reactive headroom (p.u.)
%     .aClipped [9 x 1] the action after clipping to [-1,1]
%
%   WHY A CONTINUOUS RELAXATION
%   ---------------------------
%   The true action space is hybrid: the tap is an integer, the banks are
%   binary, and the inverter setpoints are continuous. MATLAB's
%   Reinforcement Learning Toolbox has no native hybrid PPO agent. The
%   standard remedy -- used here -- is to let the policy emit a
%   continuous vector and to quantize inside the environment. The
%   quantization is part of the environment, not the policy, so PPO's
%   likelihood ratio stays well defined. The cost is a small plateau in
%   the reward landscape around each quantisation boundary, which the
%   entropy bonus (cfg.ppo.entropyWeight) is there to escape.

a = a(:);
nA = cfg.dims.nAction;
if numel(a) ~= nA
    error('decodeAction:badSize', 'Action must have %d elements, got %d.', nA, numel(a));
end

% PPO samples from an unbounded Gaussian before clipping; enforce the box.
a = max(-1, min(1, a));
dev.aClipped = a;

nCap = cfg.dims.nCap;
nPV  = cfg.dims.nPV;

% ---------------------------------------------------------------------
% 1. OLTC tap: map [-1,1] -> {tapMin ... tapMax} by rounding
% ---------------------------------------------------------------------
tapSpan  = (cfg.oltc.tapMax - cfg.oltc.tapMin) / 2;
tapMid   = (cfg.oltc.tapMax + cfg.oltc.tapMin) / 2;
dev.tap  = round(tapMid + a(1) * tapSpan);
dev.tap  = max(cfg.oltc.tapMin, min(cfg.oltc.tapMax, dev.tap));
dev.V0   = cfg.oltc.nominalPU + dev.tap * cfg.oltc.stepPU;

% ---------------------------------------------------------------------
% 2. Capacitor banks: [-1,1] -> on/off (or fractional)
% ---------------------------------------------------------------------
aCap = a(2 : 1+nCap);
if cfg.cap.binary
    dev.capOn = double(aCap > 0);
else
    dev.capOn = (aCap + 1) / 2;          % fractional energisation
end
capRatedPU = cfg.cap.kvar(:) / 1000 / cfg.net.baseMVA;
dev.capQ   = dev.capOn(:) .* capRatedPU;

% ---------------------------------------------------------------------
% 3. PV inverter reactive setpoints, limited by the apparent-power circle
% ---------------------------------------------------------------------
aInv = a(2+nCap : 1+nCap+nPV);
Srated = (cfg.pv.ratedKW(:) / 1000 / cfg.net.baseMVA) / cfg.pv.pfMin;

if nargin < 3 || isempty(pvPnow)
    pvPnow = zeros(nPV,1);
end
pvPnow = pvPnow(:);
pvPnow = min(pvPnow, Srated);            % guard against rounding overshoot

dev.invQmax = sqrt(max(0, Srated.^2 - pvPnow.^2));
dev.invQ    = aInv(:) .* dev.invQmax;

% Device bus lists, carried along so downstream code never re-derives them.
dev.capBuses = cfg.cap.buses(:);
dev.pvBuses  = cfg.pv.buses(:);
end
