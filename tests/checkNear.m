function s = checkNear(s, label, actual, expected, tol)
%CHECKNEAR  Record a numeric assertion with a tolerance.
ok = isfinite(actual) && abs(actual - expected) <= tol;
if ok
    s.pass = s.pass + 1;
    fprintf('  [pass] %s (%.6g)\n', label, actual);
else
    s.fail = s.fail + 1;
    s.failures{end+1} = sprintf('%s: got %.6g, expected %.6g +/- %.3g', ...
        label, actual, expected, tol);
    fprintf('  [FAIL] %s: got %.6g, expected %.6g +/- %.3g\n', ...
        label, actual, expected, tol);
end
end
