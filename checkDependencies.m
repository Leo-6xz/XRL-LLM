function status = checkDependencies(verbose)
%CHECKDEPENDENCIES  Report which optional components are available.
%
%   STATUS = CHECKDEPENDENCIES() prints a table and returns a struct of
%   logicals.
%
%   Only base MATLAB is REQUIRED. The pipeline degrades gracefully:
%
%     Reinforcement Learning Toolbox  missing -> use RULEPOLICY instead of a
%                                     trained agent; every other stage runs
%     Deep Learning Toolbox           missing -> same as above
%     LLMs with MATLAB add-on         missing -> template explanations
%     Ollama server                   down    -> template explanations
%     Parallel Computing Toolbox      missing -> single-worker training
%     Statistics and ML Toolbox       missing -> nothing lost; the SHAP,
%                                     correlation and t-test code here is
%                                     self-contained by design
%
%   That last point is deliberate. Depending on the Statistics Toolbox for
%   a Spearman correlation or a t-test would make the repository harder to
%   reproduce for no benefit, so those few functions are implemented
%   locally (see COUNTERFACTUALTEST and ABLATIONEXPERIMENTS).

if nargin < 1, verbose = true; end

status.matlabVersion = version('-release');
status.rlToolbox     = hasToolbox('rl',       'Reinforcement Learning Toolbox');
status.dlToolbox     = hasToolbox('nnet',     'Deep Learning Toolbox');
status.statsToolbox  = hasToolbox('stats',    'Statistics and Machine Learning Toolbox');
status.pctToolbox    = hasToolbox('parallel', 'Parallel Computing Toolbox');
status.simulink      = hasToolbox('simulink', 'Simulink');

status.llmAddon      = ~isempty(which('ollamaChat')) || ~isempty(which('openAIChat'));

status.ollamaLive = false;
if status.llmAddon && ~isempty(which('ollamaChat'))
    try
        c = ollamaChat('llama3:8b','TimeOut',5);
        r = generate(c,'Reply with OK.');
        status.ollamaLive = ~isempty(r);
    catch
        status.ollamaLive = false;
    end
end

try
    status.gpu = (gpuDeviceCount > 0);
catch
    status.gpu = false;
end

status.canTrain      = status.rlToolbox && status.dlToolbox;
status.canExplainLLM = status.llmAddon && status.ollamaLive;
status.canRunCore    = true;

if verbose
    fprintf('\n=== XRL-LLM dependency check (MATLAB %s) ===\n', status.matlabVersion);
    depLine('Core pipeline (physics, SHAP, evaluation)', status.canRunCore, 'base MATLAB only');
    depLine('Reinforcement Learning Toolbox', status.rlToolbox, 'needed to TRAIN; rulePolicy otherwise');
    depLine('Deep Learning Toolbox',          status.dlToolbox, 'needed for the policy network');
    depLine('Simulink',                       status.simulink,  'optional high-fidelity validation only');
    depLine('Parallel Computing Toolbox',     status.pctToolbox,'optional training speed-up');
    depLine('Statistics and ML Toolbox',      status.statsToolbox,'not required; stats are self-contained');
    depLine('LLMs with MATLAB add-on',        status.llmAddon,  'needed for natural-language generation');
    depLine('Ollama server reachable',        status.ollamaLive,'run: ollama serve && ollama pull llama3:8b');
    depLine('GPU available',                  status.gpu,       'optional');

    fprintf('\n  Verdict: ');
    if status.canTrain && status.canExplainLLM
        fprintf('full pipeline available.\n');
    elseif status.canTrain
        fprintf('training available; explanations will use the template backend.\n');
    elseif status.canExplainLLM
        fprintf('no RL Toolbox -- use rulePolicy; LLM explanations available.\n');
    else
        fprintf(['core only -- rulePolicy + template explanations. Everything\n' ...
                 '           still runs end to end; only the policy and the prose differ.\n']);
    end
    fprintf('\n');
end
end

function tf = hasToolbox(name, ~)
tf = ~isempty(ver(name));
end

function depLine(label, tf, note)
if tf, mark = 'yes'; else, mark = 'NO '; end
fprintf('  [%s] %-42s %s\n', mark, label, note);
end
