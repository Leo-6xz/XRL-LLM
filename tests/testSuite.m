function s = testSuite(name)
%TESTSUITE  Minimal assertion harness (no toolbox dependency).
%
%   Deliberately not built on the MATLAB unit-testing framework: this way
%   the suite runs on any installation, and the pass/fail output is plain
%   text that can be pasted into a lab notebook or a CI log.
s.name = name;
s.pass = 0;
s.fail = 0;
s.failures = {};
fprintf('\n=== %s ===\n', name);
end
