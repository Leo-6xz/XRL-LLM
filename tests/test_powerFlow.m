function results = test_powerFlow()
%TEST_POWERFLOW  Validate the load-flow solver against published results.
%
%   The base-case figures below are the standard Baran & Wu (1989) results
%   for the 33-bus feeder and are reproduced in essentially every paper
%   that uses it. If this test fails, nothing downstream means anything.

results = testSuite('powerFlow');
data   = ieee33BusData();
feeder = buildFeeder(data);

% --- Topology ---------------------------------------------------------
results = check(results,'33 buses',            feeder.nBus == 33);
results = check(results,'32 branches',         feeder.nBranch == 32);
results = check(results,'radial (tree)',       feeder.nBranch == feeder.nBus - 1);
results = check(results,'slack has no parent', feeder.parentBus(1) == 0);
results = check(results,'all buses reachable', all(isfinite(feeder.hopDist(1,:))));
results = check(results,'total load 3715 kW',  abs(sum(data.load(:,2)) - 3715) < 1e-9);
results = check(results,'total load 2300 kvar',abs(sum(data.load(:,3)) - 2300) < 1e-9);

% --- Base case against the published benchmark ------------------------
pf = radialPowerFlow(feeder, feeder.Pload0, feeder.Qload0, 1.0);

results = check(results,'converged', pf.converged);
results = check(results,'converged in < 20 iterations', pf.iterations < 20);

[vmin, ibus] = min(pf.Vm);
results = checkNear(results,'min voltage 0.9131 p.u.', vmin, 0.9131, 5e-4);
results = check(results,'min voltage at bus 18', ibus == 18);

lossKW = pf.Ploss * data.baseMVA * 1000;
results = checkNear(results,'losses 202.67 kW', lossKW, 202.67, 0.5);

lossKvar = pf.Qloss * data.baseMVA * 1000;
results = checkNear(results,'reactive losses 135.14 kvar', lossKvar, 135.14, 0.5);

results = check(results,'slack voltage held', abs(pf.Vm(1) - 1.0) < 1e-12);
results = check(results,'voltage falls downstream', pf.Vm(18) < pf.Vm(2));

% --- Physical sanity --------------------------------------------------
pfNL = radialPowerFlow(feeder, zeros(33,1), zeros(33,1), 1.0);
results = checkNear(results,'no-load: flat profile', max(abs(pfNL.Vm - 1)), 0, 1e-10);
results = checkNear(results,'no-load: zero losses',  pfNL.Ploss, 0, 1e-12);

pfCap = feeder.Qload0; pfCap(30) = pfCap(30) - 0.05;
pfC = radialPowerFlow(feeder, feeder.Pload0, pfCap, 1.0);
results = check(results,'VAr injection raises local voltage', pfC.Vm(30) > pf.Vm(30));

pfHi = radialPowerFlow(feeder, feeder.Pload0, feeder.Qload0, 1.05);
results = check(results,'higher slack raises all voltages', all(pfHi.Vm > pf.Vm));

pfHeavy = radialPowerFlow(feeder, 1.5*feeder.Pload0, 1.5*feeder.Qload0, 1.0);
results = check(results,'heavier load increases losses', pfHeavy.Ploss > pf.Ploss);
results = check(results,'heavier load lowers min voltage', min(pfHeavy.Vm) < vmin);

results = report(results);
end
