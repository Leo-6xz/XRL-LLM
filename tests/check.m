function s = check(s, label, condition)
%CHECK  Record a boolean assertion.
if condition
    s.pass = s.pass + 1;
    fprintf('  [pass] %s\n', label);
else
    s.fail = s.fail + 1;
    s.failures{end+1} = label;
    fprintf('  [FAIL] %s\n', label);
end
end
