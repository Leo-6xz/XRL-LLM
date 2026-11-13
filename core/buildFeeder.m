function feeder = buildFeeder(data)
%BUILDFEEDER  Convert raw 33-bus data into a solver-ready feeder struct.
%
%   FEEDER = BUILDFEEDER(DATA) takes the output of IEEE33BUSDATA and
%   produces the per-unit, topologically-ordered structure consumed by
%   RADIALPOWERFLOW.
%
%   The key product is FEEDER.ORDER: a permutation of branch indices such
%   that every branch appears *after* the branch feeding its from-bus.
%   This single ordering makes both sweeps of the backward/forward sweep
%   algorithm a simple loop (forward = 1:end, backward = end:-1:1), with
%   no recursion and no sparse factorisation.
%
%   Fields
%   ------
%     nBus, nBranch, baseKV, baseMVA, baseZ
%     from, to      : [nBranch x 1] bus indices
%     Z             : [nBranch x 1] complex branch impedance (p.u.)
%     R             : [nBranch x 1] branch resistance (p.u.)
%     order         : [nBranch x 1] BFS branch ordering (parent first)
%     parentBus     : [nBus x 1] upstream bus of each bus (0 for slack)
%     depth         : [nBus x 1] hop count from the substation
%     Pload0, Qload0: [nBus x 1] nominal load (p.u.)
%     pf0           : [nBus x 1] nominal power factor per bus
%     hopDist       : [nBus x nBus] shortest-path hop distance
%     zone          : [nBus x 1] lateral index (1 = main feeder, 2..4 laterals)
%     zoneName      : cellstr of zone labels
%
%   See also IEEE33BUSDATA, RADIALPOWERFLOW.

if nargin < 1 || isempty(data)
    data = ieee33BusData();
end

nBus    = data.nBus;
nBranch = data.nBranch;

feeder.nBus    = nBus;
feeder.nBranch = nBranch;
feeder.baseKV  = data.baseKV;
feeder.baseMVA = data.baseMVA;

% Impedance base: Zbase = kV^2 / MVA  (line-to-line kV, three-phase MVA)
feeder.baseZ = data.baseKV^2 / data.baseMVA;

feeder.from = data.branch(:,1);
feeder.to   = data.branch(:,2);
feeder.R    = data.branch(:,3) / feeder.baseZ;
X           = data.branch(:,4) / feeder.baseZ;
feeder.Z    = feeder.R + 1i*X;

% ---------------------------------------------------------------------
% Nominal loads in per unit (kW -> MW -> p.u.)
% ---------------------------------------------------------------------
feeder.Pload0 = data.load(:,2) / 1000 / data.baseMVA;
feeder.Qload0 = data.load(:,3) / 1000 / data.baseMVA;

% Nominal power factor per bus. Used to derive Q from P when the RL
% observation carries only active load (keeps the observation vector at
% 80 features, see BUILDOBSERVATION).
S0 = hypot(feeder.Pload0, feeder.Qload0);
feeder.pf0 = ones(nBus,1);
nz = S0 > 0;
feeder.pf0(nz) = feeder.Pload0(nz) ./ S0(nz);
% Q/P ratio, the quantity actually used at run time (finite everywhere).
feeder.qpRatio = zeros(nBus,1);
nzP = feeder.Pload0 > 0;
feeder.qpRatio(nzP) = feeder.Qload0(nzP) ./ feeder.Pload0(nzP);

% ---------------------------------------------------------------------
% Breadth-first traversal from the slack bus (bus 1) to obtain a valid
% parent-before-child branch ordering.  Also verifies radiality.
% ---------------------------------------------------------------------
adj = cell(nBus,1);
for b = 1:nBranch
    f = feeder.from(b); t = feeder.to(b);
    adj{f}(end+1,1) = b;
    adj{t}(end+1,1) = b;
end

visited   = false(nBus,1);
parentBus = zeros(nBus,1);
depth     = zeros(nBus,1);
order     = zeros(nBranch,1);

visited(1) = true;
queue      = 1;
nOrdered   = 0;

while ~isempty(queue)
    u     = queue(1);
    queue(1) = [];
    for b = adj{u}(:)'
        v = feeder.from(b) + feeder.to(b) - u;   % the other endpoint
        if visited(v)
            continue
        end
        visited(v)   = true;
        parentBus(v) = u;
        depth(v)     = depth(u) + 1;
        nOrdered     = nOrdered + 1;
        order(nOrdered) = b;
        % Normalise orientation so that `from` is always upstream.
        feeder.from(b) = u;
        feeder.to(b)   = v;
        queue(end+1) = v;  %#ok<AGROW>
    end
end

if nOrdered ~= nBranch || ~all(visited)
    error('buildFeeder:notRadial', ...
        ['Network is not a connected radial tree: visited %d of %d buses ', ...
         'using %d of %d branches. Check the branch table.'], ...
        nnz(visited), nBus, nOrdered, nBranch);
end

feeder.order     = order;
feeder.parentBus = parentBus;
feeder.depth     = depth;

% ---------------------------------------------------------------------
% All-pairs hop distance (small graph -> simple BFS per source is fine).
% Used by the interpretability penalty to judge whether two devices that
% moved together are electrically "near" each other.
% ---------------------------------------------------------------------
hopDist = inf(nBus);
for s = 1:nBus
    d = inf(nBus,1); d(s) = 0;
    q = s;
    while ~isempty(q)
        u = q(1); q(1) = [];
        for b = adj{u}(:)'
            v = feeder.from(b) + feeder.to(b) - u;
            if isinf(d(v))
                d(v) = d(u) + 1;
                q(end+1) = v; %#ok<AGROW>
            end
        end
    end
    hopDist(s,:) = d';
end
feeder.hopDist  = hopDist;
feeder.diameter = max(hopDist(:));

% ---------------------------------------------------------------------
% Electrical zones = the four natural laterals of the 33-bus feeder.
% These give the LLM prompt a human-meaningful spatial vocabulary.
% ---------------------------------------------------------------------
% Other feeders (e.g. IEEE69BUSDATA) supply their own zones.
if isfield(data, 'zone') && isfield(data, 'zoneName')
    feeder.zone     = data.zone(:);
    feeder.zoneName = data.zoneName;
else
    zone = ones(nBus,1);
    zone(19:22) = 2;
    zone(23:25) = 3;
    zone(26:33) = 4;
    feeder.zone     = zone;
    feeder.zoneName = { ...
        'Main feeder (buses 1-18)', ...
        'Lateral A off bus 2 (buses 19-22)', ...
        'Lateral B off bus 3 (buses 23-25)', ...
        'Lateral C off bus 6 (buses 26-33)'};
end
feeder.feederEndBus = 18;          % deepest bus of the main feeder
if isfield(data, 'feederEndBus'), feeder.feederEndBus = data.feederEndBus; end
end
