function data = ieee33BusData()
%IEEE33BUSDATA  Baran & Wu (1989) 33-bus radial distribution test feeder.
%
%   DATA = IEEE33BUSDATA() returns a struct with the raw network data in
%   engineering units, exactly as published in:
%
%     M. E. Baran and F. F. Wu, "Network reconfiguration in distribution
%     systems for loss reduction and load balancing," IEEE Trans. Power
%     Delivery, vol. 4, no. 2, pp. 1401-1407, 1989.
%
%   Fields
%   ------
%     baseKV     : line-to-line base voltage (kV)
%     baseMVA    : system base power (MVA)
%     branch     : [nBranch x 4] = [fromBus toBus R_ohm X_ohm]
%     load       : [nBus x 3]    = [bus P_kW Q_kvar]
%     nBus       : number of buses (33)
%     nBranch    : number of branches (32)
%
%   Sanity check: total load = 3715 kW + j2300 kvar, which matches the
%   figure quoted in the project specification. The base-case minimum
%   voltage (flat 1.0 p.u. slack, no compensation) is ~0.9131 p.u. at bus
%   18, and total losses are ~202.7 kW. TEST_POWERFLOW verifies both.
%
%   See also BUILDFEEDER, RADIALPOWERFLOW.

data.baseKV  = 12.66;   % kV, line-to-line
data.baseMVA = 10.0;    % MVA
data.nBus    = 33;

% ---------------------------------------------------------------------
% Branch data: [from  to   R (ohm)   X (ohm)]
% Branches 1-17 form the main feeder, 18-21 the lateral off bus 2,
% 22-24 the lateral off bus 3, and 25-32 the lateral off bus 6.
% ---------------------------------------------------------------------
data.branch = [ ...
     1   2   0.0922  0.0470
     2   3   0.4930  0.2511
     3   4   0.3660  0.1864
     4   5   0.3811  0.1941
     5   6   0.8190  0.7070
     6   7   0.1872  0.6188
     7   8   0.7114  0.2351
     8   9   1.0300  0.7400
     9  10   1.0440  0.7400
    10  11   0.1966  0.0650
    11  12   0.3744  0.1238
    12  13   1.4680  1.1550
    13  14   0.5416  0.7129
    14  15   0.5910  0.5260
    15  16   0.7463  0.5450
    16  17   1.2890  1.7210
    17  18   0.7320  0.5740
     2  19   0.1640  0.1565
    19  20   1.5042  1.3554
    20  21   0.4095  0.4784
    21  22   0.7089  0.9373
     3  23   0.4512  0.3083
    23  24   0.8980  0.7091
    24  25   0.8960  0.7011
     6  26   0.2030  0.1034
    26  27   0.2842  0.1447
    27  28   1.0590  0.9337
    28  29   0.8042  0.7006
    29  30   0.5075  0.2585
    30  31   0.9744  0.9630
    31  32   0.3105  0.3619
    32  33   0.3410  0.5302];

data.nBranch = size(data.branch, 1);

% ---------------------------------------------------------------------
% Load data: [bus  P (kW)  Q (kvar)].  Bus 1 is the substation (slack).
% ---------------------------------------------------------------------
data.load = [ ...
     1     0     0
     2   100    60
     3    90    40
     4   120    80
     5    60    30
     6    60    20
     7   200   100
     8   200   100
     9    60    20
    10    60    20
    11    45    30
    12    60    35
    13    60    35
    14   120    80
    15    60    10
    16    60    20
    17    60    20
    18    90    40
    19    90    40
    20    90    40
    21    90    40
    22    90    40
    23    90    50
    24   420   200
    25   420   200
    26    60    25
    27    60    25
    28    60    20
    29   120    70
    30   200   600
    31   150    70
    32   210   100
    33    60    40];

% Defensive check -- guards against silent corruption of the tables above.
assert(size(data.load, 1) == data.nBus, ...
    'ieee33BusData:badLoadTable', 'Load table must have %d rows.', data.nBus);
assert(abs(sum(data.load(:,2)) - 3715) < 1e-6, ...
    'ieee33BusData:badLoadTotal', 'Total active load must be 3715 kW.');
assert(abs(sum(data.load(:,3)) - 2300) < 1e-6, ...
    'ieee33BusData:badLoadTotal', 'Total reactive load must be 2300 kvar.');
end
