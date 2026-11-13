function s = report(s)
%REPORT  Print the suite summary.
n = s.pass + s.fail;
if s.fail == 0
    fprintf('  --> %d/%d passed\n', s.pass, n);
else
    fprintf('  --> %d/%d passed, %d FAILED:\n', s.pass, n, s.fail);
    for i = 1:numel(s.failures)
        fprintf('        - %s\n', s.failures{i});
    end
end
end
