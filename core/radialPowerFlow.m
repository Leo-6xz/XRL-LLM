function pf = radialPowerFlow(feeder, Pinj, Qinj, V0, opts)
%RADIALPOWERFLOW  Backward/forward sweep load flow for a radial feeder.
%
%   PF = RADIALPOWERFLOW(FEEDER, PINJ, QINJ, V0) solves the balanced
%   single-phase-equivalent power flow of a radial distribution network.
%
%   Inputs
%   ------
%     FEEDER : struct from BUILDFEEDER
%     PINJ   : [nBus x 1] NET active power *consumed* at each bus (p.u.).
%              Positive = load. Generation enters as a negative value.
%     QINJ   : [nBus x 1] NET reactive power consumed (p.u.). Capacitor
%              banks and inverter VAr injection enter as negative values.
%     V0     : complex or real slack voltage at bus 1 (p.u.). Scalar.
%     OPTS   : optional struct with fields TOL (default 1e-8) and
%              MAXITER (default 100).
%
%   Output PF
%   ---------
%     .V         [nBus x 1]  complex bus voltages (p.u.)
%     .Vm        [nBus x 1]  voltage magnitudes (p.u.)
%     .Ibr       [nBranch x 1] complex branch currents (p.u.)
%     .Ploss     scalar total active loss (p.u.)
%     .Qloss     scalar total reactive loss (p.u.)
%     .iterations, .converged, .mismatch
%
%   WHY THIS SOLVER AND NOT SIMULINK/SIMSCAPE
%   -----------------------------------------
%   For a *balanced radial* network the backward/forward sweep is exact
%   (it is not an approximation of Newton-Raphson -- it solves the same
%   nonlinear equations) and converges in roughly 4-8 iterations. One
%   solve costs well under a millisecond. PPO training here needs on the
%   order of 5e5 solves; at Simscape's ~0.5-2 s per episode-step that is
%   days to weeks of wall time, whereas this solver completes the same
%   sweep in minutes. Keep Simulink/Simscape for high-fidelity validation
%   of the *final* policy (see README, "Simulink validation harness"),
%   not for the training loop.
%
%   See also BUILDFEEDER, XRLSTEP.

if nargin < 5 || isempty(opts)
    opts = struct();
end
if ~isfield(opts,'tol'),     opts.tol     = 1e-8; end
if ~isfield(opts,'maxIter'), opts.maxIter = 100;  end

nBus    = feeder.nBus;
nBranch = feeder.nBranch;

Pinj = Pinj(:);
Qinj = Qinj(:);
if numel(Pinj) ~= nBus || numel(Qinj) ~= nBus
    error('radialPowerFlow:badSize', ...
        'Pinj and Qinj must both be %d-element vectors.', nBus);
end

% Complex power drawn at each bus. The slack absorbs the balance.
S = Pinj + 1i*Qinj;
S(1) = 0;

V      = V0 * ones(nBus,1);
V(1)   = V0;
Ibr    = zeros(nBranch,1);
ordr   = feeder.order;
from   = feeder.from;
to     = feeder.to;
Z      = feeder.Z;

converged = false;
mismatch  = Inf;
it        = 0;

for it = 1:opts.maxIter
    Vprev = V;

    % ---- Backward sweep -------------------------------------------
    % Bus current injections, then accumulate them up the tree. Because
    % `ordr` lists a parent before any of its children, walking it in
    % reverse guarantees every child's contribution is already folded in
    % by the time we reach the parent.
    Ibus    = conj(S ./ V);
    Ibus(1) = 0;
    Iacc    = Ibus;
    for k = nBranch:-1:1
        b = ordr(k);
        Ibr(b)       = Iacc(to(b));
        Iacc(from(b)) = Iacc(from(b)) + Iacc(to(b));
    end

    % ---- Forward sweep --------------------------------------------
    V(1) = V0;
    for k = 1:nBranch
        b = ordr(k);
        V(to(b)) = V(from(b)) - Z(b) * Ibr(b);
    end

    mismatch = max(abs(abs(V) - abs(Vprev)));
    if mismatch < opts.tol
        converged = true;
        break
    end
end

pf.V          = V;
pf.Vm         = abs(V);
pf.Ibr        = Ibr;
pf.iterations = it;
pf.converged  = converged;
pf.mismatch   = mismatch;

% Branch losses: |I|^2 * Z
Isq      = abs(Ibr).^2;
lossVec  = Isq .* Z;
pf.Ploss = sum(real(lossVec));
pf.Qloss = sum(imag(lossVec));
pf.branchLoss = real(lossVec);

if ~converged
    warning('radialPowerFlow:noConvergence', ...
        ['Power flow did not converge in %d iterations (mismatch %.3e). ', ...
         'This usually means the injected load is far outside the ', ...
         'feeder''s capability -- check the scenario scaling.'], ...
        opts.maxIter, mismatch);
end
end
