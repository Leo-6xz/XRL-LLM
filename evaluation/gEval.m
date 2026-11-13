function results = gEval(explanations, records, llm, cfg, opts)
%GEVAL  LLM-as-judge scoring of explanation quality.
%
%   RESULTS = GEVAL(EXPLANATIONS, RECORDS, LLM, CFG, OPTS)
%
%   Scores each explanation 1-5 on the three dimensions named in the brief:
%     ACCURACY       does it faithfully reflect the decision logic?
%     ACTIONABILITY  does it give the operator something to act on?
%     COMPLETENESS   does it cover the key contributing factors?
%
%   EXPLANATIONS is a cellstr; RECORDS is the matching record array (the
%   judge needs the ground-truth state and attribution to score accuracy,
%   otherwise it is scoring fluency).
%
%   See also GENERATEEXPLANATION, ABLATIONEXPERIMENTS.

if nargin < 5 || isempty(opts), opts = struct(); end
if ~isfield(opts,'verbose'), opts.verbose = cfg.io.verbose; end
if ~isfield(opts,'maxN'),    opts.maxN    = cfg.eval.nGEvalSamples; end

if ischar(explanations) || isstring(explanations)
    explanations = cellstr(explanations);
end

n = min(numel(explanations), opts.maxN);
if n == 0
    error('gEval:empty', 'No explanations to score.');
end

useLLM = llm.available;
if useLLM && strcmp(cfg.llm.evalModel, cfg.llm.model)
    warning('gEval:selfJudge', ...
        ['The judge model (%s) is the same as the generator model. LLM judges ', ...
         'systematically favour their own output. Set cfg.llm.evalModel to a ', ...
         'different model before reporting these scores.'], cfg.llm.evalModel);
end

acc = nan(n,1); act = nan(n,1); comp = nan(n,1);
raw = cell(n,1);

for i = 1:n
    if useLLM
        try
            [acc(i), act(i), comp(i), raw{i}] = judgeWithLLM( ...
                explanations{i}, records(min(i,numel(records))), llm, cfg);
        catch ME
            warning('gEval:judgeFailed', ...
                'Judge failed on item %d (%s); using the proxy scorer.', i, ME.message);
            [acc(i), act(i), comp(i)] = proxyScore( ...
                explanations{i}, records(min(i,numel(records))), cfg);
            raw{i} = 'proxy';
        end
    else
        [acc(i), act(i), comp(i)] = proxyScore( ...
            explanations{i}, records(min(i,numel(records))), cfg);
        raw{i} = 'proxy';
    end

    if opts.verbose && mod(i,20) == 0
        fprintf('  G-Eval: %d/%d scored\n', i, n);
    end
end

results.n              = n;
results.accuracy       = acc;
results.actionability  = act;
results.completeness   = comp;
results.overall        = mean([acc act comp], 2);

results.meanAccuracy      = mean(acc, 'omitnan');
results.meanActionability = mean(act, 'omitnan');
results.meanCompleteness  = mean(comp, 'omitnan');
results.meanOverall       = mean(results.overall, 'omitnan');
results.sdOverall         = std(results.overall, 'omitnan');

% Ceiling detection. A condition whose scores are all at or near the top
% of the scale tells you the rubric has no headroom left, and any ranking
% among the top conditions is then meaningless. This fires in practice:
% a well-constructed template hits every rubric clause mechanically and
% saturates at 5.00.
results.atCeiling = (results.meanOverall >= 4.9) || (results.sdOverall < 0.05);

results.proxy        = ~useLLM;
results.judgeModel   = cfg.llm.evalModel;
results.passesTarget = results.meanOverall >= cfg.eval.targetQuality;
results.rawResponses = raw;

if opts.verbose
    fprintf('\n--- Explanation quality (%s) ---\n', ...
        ternary(results.proxy, 'PROXY RUBRIC, not G-Eval', ...
        sprintf('G-Eval, judge = %s', results.judgeModel)));
    fprintf('  Accuracy      : %.2f / 5\n', results.meanAccuracy);
    fprintf('  Actionability : %.2f / 5\n', results.meanActionability);
    fprintf('  Completeness  : %.2f / 5\n', results.meanCompleteness);
    fprintf('  Overall       : %.2f +/- %.2f  (target >= %.1f) %s\n', ...
        results.meanOverall, results.sdOverall, cfg.eval.targetQuality, ...
        ternary(results.passesTarget, '[PASS]', '[BELOW TARGET]'));
    if results.proxy
        fprintf(['  NOTE: no LLM judge was reachable. These are rubric proxy\n' ...
                 '  scores and must not be reported as G-Eval results.\n']);
    end
    if results.atCeiling
        fprintf(['  WARNING: scores are at the ceiling (mean %.2f, SD %.2f). The\n' ...
                 '  rubric has no headroom, so any ranking among top conditions is\n' ...
                 '  not meaningful. Add discriminating dimensions before reporting.\n'], ...
                 results.meanOverall, results.sdOverall);
    end
end
end

% =====================================================================
function [a, ac, co, raw] = judgeWithLLM(expl, record, llm, cfg)
%JUDGEWITHLLM  Score one explanation against the ground-truth record.

Vm = record.state.Vm(:);
gt = sprintf([ ...
    'Lowest voltage %.4f p.u. at bus %d; highest %.4f p.u. at bus %d; ' ...
    '%d bus(es) outside the %.2f-%.2f p.u. band. Action actually taken: %s ' ...
    'Top attributed factor: %s (%.0f%% of total attribution).'], ...
    min(Vm), find(Vm==min(Vm),1), max(Vm), find(Vm==max(Vm),1), ...
    nnz(Vm < cfg.limits.Vmin | Vm > cfg.limits.Vmax), ...
    cfg.limits.Vmin, cfg.limits.Vmax, record.actionText, ...
    record.topFeatures(1).name, 100*record.topFeatures(1).share);

prompt = sprintf([ ...
    'You are evaluating an explanation produced by an AI assistant for a power\n' ...
    'system control decision. You are given the GROUND TRUTH and the EXPLANATION.\n\n' ...
    '[GROUND TRUTH]\n%s\n\n' ...
    '[EXPLANATION TO EVALUATE]\n%s\n\n' ...
    '[SCORING RUBRIC - integers 1 to 5]\n' ...
    'accuracy: 5 = every factual claim matches the ground truth and the stated\n' ...
    '  cause matches the top attributed factor; 3 = broadly right with one\n' ...
    '  imprecision; 1 = contains a claim contradicted by the ground truth, or\n' ...
    '  invents a number or bus that does not appear above.\n' ...
    'actionability: 5 = an operator could act or decide not to act on this alone;\n' ...
    '  3 = informative but the operational implication is left implicit;\n' ...
    '  1 = describes the action without any operational consequence.\n' ...
    'completeness: 5 = covers the trigger condition, the decisive factor and the\n' ...
    '  expected effect; 3 = covers two of the three; 1 = covers one.\n\n' ...
    'Penalise fabricated numbers heavily under accuracy. Reward an explanation\n' ...
    'that correctly states there was no single dominant factor when that is true.\n\n' ...
    'Respond with ONLY a JSON object, no code fences and no commentary:\n' ...
    '{"accuracy": <int>, "actionability": <int>, "completeness": <int>}'], ...
    gt, expl);

raw = char(generate(llm.chat, prompt));

s = parseScoreJSON(raw);
a  = s.accuracy;
ac = s.actionability;
co = s.completeness;
end

% =====================================================================
function s = parseScoreJSON(raw)
%PARSESCOREJSON  Tolerant JSON extraction.
%   Small models routinely wrap JSON in prose or code fences despite being
%   told not to, so extract the first {...} block rather than trusting the
%   whole reply to parse.
txt = regexprep(raw, '```[a-zA-Z]*', '');
txt = strrep(txt, '```', '');
m = regexp(txt, '\{.*?\}', 'match', 'once');
if isempty(m)
    error('gEval:noJSON', 'Judge reply contained no JSON object: %s', raw);
end
s = jsondecode(m);

fn = {'accuracy','actionability','completeness'};
for i = 1:numel(fn)
    if ~isfield(s, fn{i})
        error('gEval:missingField', 'Judge reply lacked field "%s".', fn{i});
    end
    v = double(s.(fn{i}));
    s.(fn{i}) = max(1, min(5, v));
end
end

% =====================================================================
function [a, ac, co] = proxyScore(expl, record, cfg)
%PROXYSCORE  Deterministic rubric fallback. NOT G-Eval.
%
%   Four checkable properties stand in for the three judged dimensions.
%   This is transparent about being a proxy rather than pretending to
%   approximate a language model's judgement.

e = lower(expl);
Vm = record.state.Vm(:);

% --- accuracy: names the top factor, quotes real numbers, invents none --
topTok = regexp(lower(record.topFeatures(1).name), '[a-z]{4,}', 'match');
namesTop = ~isempty(topTok) && any(cellfun(@(t) contains(e, t), topTok));

% Every 3-4 decimal figure in the text should trace to the record.
% The admissible set must include the PREDICTED VOLTAGE DELTAS from the
% sensitivity block, not just the bus voltages: the explanation's
% "expected effect" clause quotes them, and an earlier version of this
% scorer counted every one as a fabrication and drove accuracy to 1.5/5.
quoted = regexp(expl, '\d\.\d{3,4}', 'match');
realVals = [Vm(:); record.info.Vmin; record.info.Vmax];
if isfield(record,'dev') && isfield(record.dev,'V0')
    realVals(end+1) = record.dev.V0;
end
if isfield(record,'devPrev') && ~isempty(record.devPrev) && isfield(record.devPrev,'V0')
    realVals(end+1) = record.devPrev.V0;
end
if isfield(record,'sensitivity') && isfield(record,'info')
    S  = record.sensitivity;
    bd = record.info.interpBreakdown;
    for kk = find(bd.moved(:))'
        if kk == 1
            realVals = [realVals; abs(S.dVdT(:) * (record.dev.tap - tapSafe(record.devPrev)))]; %#ok<AGROW>
        elseif (kk-1) <= size(S.dVdQ,2)
            realVals = [realVals; abs(S.dVdQ(:, kk-1) * bd.dQ(kk))]; %#ok<AGROW>
        end
    end
end
realVals = [realVals(:); -realVals(:)];
fabricated = 0;
for i = 1:numel(quoted)
    v = str2double(quoted{i});
    if ~any(abs(realVals - v) < 5e-4)
        fabricated = fabricated + 1;
    end
end

a = 3 + 1.5*namesTop - 1.5*min(2, fabricated);
a = max(1, min(5, a));

% --- actionability: states an effect, a direction, or a margin ----------
hasDirection = any(cellfun(@(w) contains(e,w), ...
    {'raise','rais','lower','increase','decrease','lift','reduce','support','absorb'}));
hasEffect    = any(cellfun(@(w) contains(e,w), ...
    {'expected','will','predicted','margin','risk','p.u.'}));
ac = 2 + 1.5*hasDirection + 1.5*hasEffect;
ac = max(1, min(5, ac));

% --- completeness: trigger + driver + effect, and length in band --------
hasTrigger = any(cellfun(@(w) contains(e,w), ...
    {'because','below','above','limit','violation','approaching','band'}));
nWords = numel(strsplit(strtrim(expl)));
lengthOK = nWords >= 15 && nWords <= 220;
co = 1 + 1.4*hasTrigger + 1.3*namesTop + 1.3*hasEffect;
if ~lengthOK
    co = co - 1;
end
co = max(1, min(5, co));
end

function t = tapSafe(devPrev)
if isempty(devPrev) || ~isfield(devPrev,'tap'), t = 0; else, t = devPrev.tap; end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
