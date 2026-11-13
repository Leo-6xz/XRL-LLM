function data = ieee69BusData()
%IEEE69BUSDATA  Baran & Wu (1989) 69-bus radial distribution test feeder.
%
%   DATA = IEEE69BUSDATA() returns the same struct layout as IEEE33BUSDATA
%   (engineering units), plus the spatial vocabulary the explanation layer
%   needs on a new feeder (zones and SHAP group definitions).
%
%   Source:
%     M. E. Baran and F. F. Wu, "Optimal capacitor placement on radial
%     distribution systems," IEEE Trans. Power Del., vol. 4, no. 1,
%     pp. 725-734, Jan. 1989, doi: 10.1109/61.19265.
%   Branch and load values transcribed from MATPOWER case69.m (v2, which
%   cites the same source; R, X in ohm; loads in kW/kvar).
%
%   Sanity check (flat 1.0 p.u. slack, no compensation, verified with an
%   independent backward/forward sweep): total load 3802.1 kW + j2694.7
%   kvar, losses 224.99 kW, minimum voltage 0.90919 p.u. at bus 65.
%
%   Topology: main feeder 1-27; laterals 28-35 and 36-46 off bus 3,
%   47-50 off bus 4, 51-52 off bus 8, 53-65 off bus 9 (the long, heavily
%   loaded lateral), 66-67 off bus 11, 68-69 off bus 12.
%
%   See also IEEE33BUSDATA, XRLCONFIG69, BUILDFEEDER.

data.baseKV  = 12.66;   % kV, line-to-line
data.baseMVA = 10.0;    % MVA
data.nBus    = 69;
data.nBranch = 68;

%        from to   R(ohm)  X(ohm)
data.branch = [
     1   2   0.0005  0.0012
     2   3   0.0005  0.0012
     3   4   0.0015  0.0036
     4   5   0.0251  0.0294
     5   6   0.3660  0.1864
     6   7   0.3810  0.1941
     7   8   0.0922  0.0470
     8   9   0.0493  0.0251
     9  10   0.8190  0.2707
    10  11   0.1872  0.0619
    11  12   0.7114  0.2351
    12  13   1.0300  0.3400
    13  14   1.0440  0.3400
    14  15   1.0580  0.3496
    15  16   0.1966  0.0650
    16  17   0.3744  0.1238
    17  18   0.0047  0.0016
    18  19   0.3276  0.1083
    19  20   0.2106  0.0690
    20  21   0.3416  0.1129
    21  22   0.0140  0.0046
    22  23   0.1591  0.0526
    23  24   0.3463  0.1145
    24  25   0.7488  0.2475
    25  26   0.3089  0.1021
    26  27   0.1732  0.0572
     3  28   0.0044  0.0108
    28  29   0.0640  0.1565
    29  30   0.3978  0.1315
    30  31   0.0702  0.0232
    31  32   0.3510  0.1160
    32  33   0.8390  0.2816
    33  34   1.7080  0.5646
    34  35   1.4740  0.4873
     3  36   0.0044  0.0108
    36  37   0.0640  0.1565
    37  38   0.1053  0.1230
    38  39   0.0304  0.0355
    39  40   0.0018  0.0021
    40  41   0.7283  0.8509
    41  42   0.3100  0.3623
    42  43   0.0410  0.0478
    43  44   0.0092  0.0116
    44  45   0.1089  0.1373
    45  46   0.0009  0.0012
     4  47   0.0034  0.0084
    47  48   0.0851  0.2083
    48  49   0.2898  0.7091
    49  50   0.0822  0.2011
     8  51   0.0928  0.0473
    51  52   0.3319  0.1140
     9  53   0.1740  0.0886
    53  54   0.2030  0.1034
    54  55   0.2842  0.1447
    55  56   0.2813  0.1433
    56  57   1.5900  0.5337
    57  58   0.7837  0.2630
    58  59   0.3042  0.1006
    59  60   0.3861  0.1172
    60  61   0.5075  0.2585
    61  62   0.0974  0.0496
    62  63   0.1450  0.0738
    63  64   0.7105  0.3619
    64  65   1.0410  0.5302
    11  66   0.2012  0.0611
    66  67   0.0047  0.0014
    12  68   0.7394  0.2444
    68  69   0.0047  0.0016];

%        bus  P(kW)  Q(kvar)
data.load = [
     1       0       0
     2       0       0
     3       0       0
     4       0       0
     5       0       0
     6     2.6     2.2
     7    40.4      30
     8      75      54
     9      30      22
    10      28      19
    11     145     104
    12     145     104
    13       8     5.5
    14       8     5.5
    15       0       0
    16    45.5      30
    17      60      35
    18      60      35
    19       0       0
    20       1     0.6
    21     114      81
    22     5.3     3.5
    23       0       0
    24      28      20
    25       0       0
    26      14      10
    27      14      10
    28      26    18.6
    29      26    18.6
    30       0       0
    31       0       0
    32       0       0
    33      14      10
    34    19.5      14
    35       6       4
    36      26    18.6
    37      26    18.6
    38       0       0
    39      24      17
    40      24      17
    41     1.2       1
    42       0       0
    43       6     4.3
    44       0       0
    45    39.2    26.3
    46    39.2    26.3
    47       0       0
    48      79    56.4
    49   384.7   274.5
    50   384.7   274.5
    51    40.5    28.3
    52     3.6     2.7
    53     4.3     3.5
    54    26.4      19
    55      24    17.2
    56       0       0
    57       0       0
    58       0       0
    59     100      72
    60       0       0
    61    1244     888
    62      32      23
    63       0       0
    64     227     162
    65      59      42
    66      18      13
    67      18      13
    68      28      20
    69      28      20];

% ---------------------------------------------------------------------
% Electrical zones (used by the LLM prompt and the dashboard)
% ---------------------------------------------------------------------
zone = ones(data.nBus, 1);
zone(28:46) = 2;
zone(47:50) = 3;
zone(51:52) = 4;
zone(53:65) = 5;
zone(66:69) = 6;
data.zone = zone;
data.zoneName = { ...
    'Main feeder (buses 1-27)', ...
    'Laterals off bus 3 (buses 28-46)', ...
    'Lateral off bus 4 (buses 47-50)', ...
    'Lateral off bus 8 (buses 51-52)', ...
    'Long lateral off bus 9 (buses 53-65)', ...
    'Laterals off buses 11-12 (buses 66-69)'};

% Lowest-voltage point of the uncontrolled feeder (end of the long lateral)
data.feederEndBus = 65;

% ---------------------------------------------------------------------
% SHAP groups: the same 12-player structure as the 33-bus study
% (6 spatial voltage bands + 2 load blocks + 4 context groups), so the
% exact enumeration stays at 2^12 = 4096 coalitions. Bus 1 (slack) is
% excluded, as on the 33-bus feeder.
% ---------------------------------------------------------------------
data.shapVoltageGroups = { ...
    'Head-end voltages (bus 2-9, 51-52)',        [2:9 51 52]; ...
    'Mid-feeder voltages (bus 10-18, 66-69)',    [10:18 66:69]; ...
    'Main feeder-end voltages (bus 19-27)',      19:27; ...
    'Head-end lateral voltages (bus 28-50)',     28:50; ...
    'Long lateral voltages (bus 53-60)',         53:60; ...
    'Long lateral end voltages (bus 61-65)',     61:65};
data.shapLoadGroups = { ...
    'Main-feeder load (bus 2-27, 51-52, 66-69)', [2:27 51 52 66:69]; ...
    'Lateral load (bus 28-50, 53-65)',           [28:50 53:65]};

% ---------------------------------------------------------------------
% Consistency checks
% ---------------------------------------------------------------------
assert(size(data.branch,1) == data.nBranch, ...
    'ieee69BusData:badBranchTable', 'Branch table must have %d rows.', data.nBranch);
assert(size(data.load,1) == data.nBus, ...
    'ieee69BusData:badLoadTable', 'Load table must have %d rows.', data.nBus);
assert(abs(sum(data.load(:,2)) - 3802.1) < 0.5, ...
    'ieee69BusData:badLoadTotal', 'Total active load must be 3802.1 kW.');
assert(abs(sum(data.load(:,3)) - 2694.7) < 0.5, ...
    'ieee69BusData:badLoadTotal', 'Total reactive load must be 2694.7 kvar.');
vb = sort([data.shapVoltageGroups{:,2}]);
lb = sort([data.shapLoadGroups{:,2}]);
assert(isequal(vb, 2:data.nBus) && isequal(lb, 2:data.nBus), ...
    'ieee69BusData:groups', 'SHAP groups must cover buses 2..69 exactly once.');
end
