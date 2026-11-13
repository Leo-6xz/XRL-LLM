function ok = runAllTests()
%RUNALLTESTS  Run every component test in dependency order.
%
%   OK = RUNALLTESTS() returns true when every assertion passed.
%
%   Order matters. The power-flow suite runs first because every other
%   result is meaningless if the physics is wrong, and a failure there
%   should stop you from debugging the SHAP layer for a problem that is
%   actually in the solver.

setupPaths();

suites = {@test_powerFlow, @test_environment, @test_reward, @test_shap, @test_llm};
names  = {'powerFlow','environment','reward','shap','llm'};

totalPass = 0; totalFail = 0; failedSuites = {};

for i = 1:numel(suites)
    try
        r = suites{i}();
        totalPass = totalPass + r.pass;
        totalFail = totalFail + r.fail;
        if r.fail > 0
            failedSuites{end+1} = names{i}; %#ok<AGROW>
        end
    catch ME
        fprintf('\n  [ERROR] suite "%s" threw: %s\n', names{i}, ME.message);
        if ~isempty(ME.stack)
            fprintf('          at %s line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        totalFail = totalFail + 1;
        failedSuites{end+1} = names{i}; %#ok<AGROW>
    end
end

fprintf('\n========================================\n');
fprintf('  TOTAL: %d passed, %d failed\n', totalPass, totalFail);
if totalFail == 0
    fprintf('  All suites passed.\n');
else
    fprintf('  Failing suites: %s\n', strjoin(failedSuites, ', '));
end
fprintf('========================================\n\n');

ok = (totalFail == 0);
end
