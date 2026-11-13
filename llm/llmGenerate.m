function [txt, meta] = llmGenerate(llm, prompt, record, cfg, feeder, level, variant)
%LLMGENERATE  Single entry point for text generation, with graceful fallback.
%
%   [TXT, META] = LLMGENERATE(LLM, PROMPT, RECORD, CFG, FEEDER, LEVEL)
%
%   Routes to the connected backend, or to the deterministic template
%   generator when no model is reachable. RECORD/CFG/FEEDER/LEVEL are only
%   needed by the template path and may be omitted when a live LLM is
%   guaranteed.
%
%   META reports .backend, .latencySec and .fallback so that the latency
%   benchmark (target < 2 s) measures what actually happened rather than
%   what was configured.

t0 = tic;
meta = struct('backend', llm.backend, 'fallback', false, 'latencySec', 0);

if llm.available
    try
        txt = generate(llm.chat, prompt);
        txt = stripPreamble(txt);
        if isempty(txt)
            error('llmGenerate:empty', 'Model returned an empty string.');
        end
        meta.latencySec = toc(t0);
        return
    catch ME
        warning('llmGenerate:failed', ...
            'LLM call failed (%s); falling back to the template generator.', ME.message);
        meta.fallback = true;
    end
end

if nargin < 6, level = 'standard'; end
if nargin < 7, variant = 'full'; end
if nargin < 5 || isempty(feeder), feeder = buildFeeder(cfg.net.dataFcn()); end

txt = generateTemplateExplanation(record, cfg, feeder, level, variant);
meta.backend    = 'template';
meta.latencySec = toc(t0);
end
