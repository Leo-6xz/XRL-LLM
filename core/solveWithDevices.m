function [pf, Pinj, Qinj] = solveWithDevices(state, dev, cfg, feeder)
%SOLVEWITHDEVICES  Assemble net bus injections and run the power flow.
%
%   [PF, PINJ, QINJ] = SOLVEWITHDEVICES(STATE, DEV, CFG, FEEDER)
%
%   Sign convention: RADIALPOWERFLOW consumes *load*, so a positive entry
%   in PINJ/QINJ is power drawn from the network. PV active generation and
%   capacitive/inverter VAr injection therefore enter with a minus sign.
%   Getting this backwards is the single most common bug in a first VVC
%   implementation -- TEST_ENVIRONMENT checks the sign explicitly by
%   asserting that closing a capacitor bank raises the local voltage.
%
%   See also RADIALPOWERFLOW, DECODEACTION, XRLSTEP.

nBus = feeder.nBus;

Pinj = state.Pload(:);
Qinj = state.Qload(:);

% --- PV inverters: active generation and commanded reactive support ---
for k = 1:numel(cfg.pv.buses)
    b = cfg.pv.buses(k);
    Pinj(b) = Pinj(b) - state.pvP(k);
    Qinj(b) = Qinj(b) - dev.invQ(k);
end

% --- Switched capacitor banks ----------------------------------------
for k = 1:numel(cfg.cap.buses)
    b = cfg.cap.buses(k);
    Qinj(b) = Qinj(b) - dev.capQ(k);
end

assert(numel(Pinj) == nBus && numel(Qinj) == nBus, ...
    'solveWithDevices:injSize', 'Injection vectors have the wrong length.');

pf = radialPowerFlow(feeder, Pinj, Qinj, dev.V0);
end
