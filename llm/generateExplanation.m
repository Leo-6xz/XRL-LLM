function [expl, meta] = generateExplanation(record, llm, cfg, feeder, level, variant)
%GENERATEEXPLANATION  Produce one natural-language justification.
%
%   [EXPL, META] = GENERATEEXPLANATION(RECORD, LLM, CFG, FEEDER, LEVEL, VARIANT)
%
%   LEVEL   'summary' | 'standard' (default) | 'detailed'
%   VARIANT 'full' (default) | 'noshap' | 'nodomain' | 'shaponly'
%
%   META reports the backend used, the end-to-end latency and the prompt
%   itself. The prompt is retained deliberately: an explanation that cannot
%   be traced back to the exact context that produced it is not auditable,
%   and auditability is the entire premise of deploying this in a control
%   room.
%
%   See also PROMPTTEMPLATES, LLMGENERATE, GENERATELAYEREDEXPLANATIONS.

if nargin < 5 || isempty(level),   level   = 'standard'; end
if nargin < 6 || isempty(variant), variant = 'full';     end

t0 = tic;

if strcmpi(variant, 'shaponly')
    % Ablation 4's control: raw numbers, no generation at all. This is the
    % condition the brief reports a +67.9% quality gain against.
    expl = rawShapText(record);
    meta = struct('backend','raw-shap','latencySec',toc(t0), ...
                  'fallback',false,'prompt','','level',level,'variant',variant);
    return
end

prompt = promptTemplates(record, cfg, feeder, level, variant);
[expl, gmeta] = llmGenerate(llm, prompt, record, cfg, feeder, level, variant);

meta          = gmeta;
meta.prompt   = prompt;
meta.level    = level;
meta.variant  = variant;
meta.latencySec = toc(t0);
meta.nPromptChars = numel(prompt);
end

% =====================================================================
function s = rawShapText(record)
lines = {sprintf('Action: %s', record.actionText)};
lines{end+1} = sprintf('SHAP attribution for the %s channel:', ...
    record.shapMeta.explainedChannelName);
for i = 1:numel(record.topFeatures)
    tf = record.topFeatures(i);
    lines{end+1} = sprintf('  %s = %+.4f', tf.name, tf.value); %#ok<AGROW>
end
s = strjoin(lines, sprintf('\n'));
end
