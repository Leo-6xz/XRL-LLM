function top = findTopFeatures(phi, k, names, noiseFloor)
%FINDTOPFEATURES  Rank attributions by absolute magnitude.
%
%   TOP = FINDTOPFEATURES(PHI, K, NAMES) returns a struct array of the K
%   most influential players for a single explained decision.
%
%   Fields: .idx .name .value .absValue .share .direction
%
%   .share is the fraction of TOTAL absolute attribution carried by that
%   player. It is the number worth putting in front of an operator: an
%   attribution of 0.42 means nothing on its own, whereas "this single
%   factor accounts for 61% of the decision" does. It is also what
%   separates a confident explanation from a diffuse one -- when the top
%   share is below roughly 0.25 the decision has no dominant driver and
%   the LLM should be told to say so rather than inventing a story.

phi = phi(:);
if nargin < 2 || isempty(k), k = 5; end
% Attributions below the numerical noise floor are rounding, not signal.
% Zeroing them keeps them out of the "top factors" list and out of the LLM
% prompt, where a spurious 1e-8 ranked first would produce a confident
% explanation of nothing.
if nargin >= 4 && ~isempty(noiseFloor)
    phi(abs(phi) < noiseFloor) = 0;
end
if nargin < 3 || isempty(names)
    names = arrayfun(@(i) sprintf('f%d', i), 1:numel(phi), 'UniformOutput', false);
end

total = sum(abs(phi));
if total <= 0
    total = eps;
end

[~, idx] = sort(abs(phi), 'descend');
k = min(k, numel(phi));

top = struct('idx',{},'name',{},'value',{},'absValue',{},'share',{},'direction',{});
for i = 1:k
    j = idx(i);
    top(i).idx      = j;
    top(i).name     = names{j};
    top(i).value    = phi(j);
    top(i).absValue = abs(phi(j));
    top(i).share    = abs(phi(j)) / total;
    if phi(j) >= 0
        top(i).direction = 'increased';
    else
        top(i).direction = 'decreased';
    end
end
end
