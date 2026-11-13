function ok = checkTransfer69()
%CHECKTRANSFER69  Fast sanity checks before the 69-bus transfer run (< 1 min).
%
%   OK = CHECKTRANSFER69() verifies, without training anything:
%     1. the 33-bus pipeline is unchanged by the feeder-generic edits
%        (80-D observation, identical SHAP group indices, 202.67 kW base loss);
%     2. the 69-bus data reproduce the published base case
%        (224.99 kW loss, V_min 0.9092 p.u. at bus 65);
%     3. the 69-bus observation, normalisation and SHAP groups are consistent;
%     4. the rule baseline runs closed-loop on the 69-bus feeder.
%   Run runAllTests afterwards as well; it must still report 115/115.
%
%   See also RUNTRANSFER69, XRLCONFIG69.

setupPaths();
ok = true;
fprintf('\n=== Transfer sanity checks ===\n');

% ---------------------------------------------------------------- 33-bus
c33 = xrlConfig();
f33 = buildFeeder(c33.net.dataFcn());
m33 = featureMapping(c33);
expect = {2:6, 7:13, 14:18, 19:22, 23:25, 26:33, 34:49, 50:65, 66:68, 69, 70:71, 72:80};
same = numel(m33.groupMembers) == 12 && ...
       all(cellfun(@(a,b) isequal(a(:)', b(:)'), m33.groupMembers, expect));
ok = report(ok, '33-bus: SHAP group indices unchanged', same);
[o33, s33] = xrlReset(c33, f33, RandStream('mt19937ar','Seed',1));
ok = report(ok, '33-bus: observation is 80-D', numel(o33) == 80);
ok = report(ok, '33-bus: obs(1:33) are bus voltages', max(abs(o33(1:33) - s33.Vm(:))) < 1e-12);
pf = radialPowerFlow(f33, f33.Pload0, f33.Qload0, 1.0);
ok = report(ok, sprintf('33-bus: base loss %.2f kW (expect 202.67)', pf.Ploss*10*1000), ...
            abs(pf.Ploss*10*1000 - 202.67) < 0.5);

% ---------------------------------------------------------------- 69-bus
cfg = xrlConfig69();
feeder = buildFeeder(cfg.net.dataFcn());
pf = radialPowerFlow(feeder, feeder.Pload0, feeder.Qload0, 1.0);
[vmin, imin] = min(pf.Vm);
lossKW = pf.Ploss * cfg.net.baseMVA * 1000;
ok = report(ok, sprintf('69-bus: base loss %.2f kW (expect 224.99)', lossKW), abs(lossKW - 224.99) < 0.5);
ok = report(ok, sprintf('69-bus: V_min %.4f p.u. at bus %d (expect 0.9092 at 65)', vmin, imin), ...
            abs(vmin - 0.9092) < 5e-4 && imin == 65);

map = featureMapping(cfg);
ok = report(ok, sprintf('69-bus: %d fine SHAP players (expect 12)', map.nGroup), map.nGroup == 12);
ok = report(ok, sprintf('69-bus: %d coarse SHAP players (expect 5)', map.coarse.nGroup), map.coarse.nGroup == 5);
cover = sort([map.groupMembers{:}]);
ok = report(ok, '69-bus: every feature except V1 in exactly one group', isequal(cover, 2:cfg.dims.nObs));

[obs, st] = xrlReset(cfg, feeder, RandStream('mt19937ar','Seed',1));
ok = report(ok, sprintf('69-bus: observation is %d-D (expect 152)', numel(obs)), numel(obs) == 152);
ok = report(ok, '69-bus: obs(1:69) are bus voltages', max(abs(obs(1:69) - st.Vm(:))) < 1e-12);
ok = report(ok, '69-bus: PV channels match state.pvP', max(abs(obs(138:140) - st.pvP(:))) < 1e-12);
nm = observationNormalization(cfg, feeder);
ok = report(ok, '69-bus: normalisation is 152-D and finite', ...
            numel(nm.scale) == 152 && all(isfinite(nm.scale)) && all(isfinite(nm.bias)));

fprintf('  Rule baseline, 5 episodes on the 69-bus feeder:\n');
rule = makePolicyFcn('rule', cfg);
r = controlMetrics(rule, cfg, feeder, 5, struct('label','rule (69-bus)','seed',cfg.seed+1000,'verbose',true));
ok = report(ok, '69-bus: closed-loop rollout finite', isfinite(r.meanLossKW) && isfinite(r.violationRate));

if ok
    fprintf('\nALL CHECKS PASSED. Next: runTransfer69   (Ollama stopped; ~15-25 min)\n');
else
    fprintf('\nSOME CHECKS FAILED -- do not start training; send the output.\n');
end
end

function ok = report(ok, name, pass)
if pass, tag = 'PASS'; else, tag = 'FAIL'; ok = false; end
fprintf('  [%s] %s\n', tag, name);
end
