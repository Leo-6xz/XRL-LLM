function txt = stripPreamble(txt)
%STRIPPREAMBLE  Remove conversational scaffolding from an LLM reply.
%
%   Small instruction-tuned models routinely ignore "output only the
%   explanation text" and open with "Here is the explanation:" or
%   "Here's an explanation of the controller's action:". Left in, this
%   costs three ways: it wastes generation tokens (and therefore latency),
%   it displaces content inside the length budget, and it reads as a
%   chatbot rather than an operator tool in a control room.
%
%   Stripping it in post-processing is more reliable than prompting harder.

txt = strtrim(char(txt));
if isempty(txt), return; end

patterns = { ...
    '^\s*here\s+is\s+(the|an|my)?\s*[^:\n]{0,60}:\s*', ...
    '^\s*here''?s\s+(the|an|my)?\s*[^:\n]{0,60}:\s*', ...
    '^\s*(sure|certainly|of course)[,!.]?\s*', ...
    '^\s*explanation\s*:\s*', ...
    '^\s*answer\s*:\s*', ...
    '^\s*\*\*[^*]{0,60}\*\*\s*:?\s*'};

changed = true;
guard = 0;
while changed && guard < 5
    changed = false; guard = guard + 1;
    for i = 1:numel(patterns)
        newTxt = regexprep(txt, patterns{i}, '', 'ignorecase', 'once');
        if ~strcmp(newTxt, txt)
            txt = strtrim(newTxt);
            changed = true;
        end
    end
end

% Drop markdown fences and leading bullets.
txt = regexprep(txt, '^```[a-zA-Z]*\s*', '');
txt = regexprep(txt, '\s*```$', '');
txt = regexprep(txt, '^\s*[-*]\s+', '');
txt = strtrim(txt);

if ~isempty(txt)
    txt(1) = upper(txt(1));
end
end
