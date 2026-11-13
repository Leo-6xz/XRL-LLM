function cfg = mergeConfigDefaults(cfg)

%MERGECONFIGDEFAULTS  Fill fields absent from an OLD saved config.
%

d = xrlConfig();
cfg = mergeInto(cfg, d);
end

function a = mergeInto(a, d)
fn = fieldnames(d);
for i = 1:numel(fn)
    f = fn{i};
    if ~isfield(a, f)
        a.(f) = d.(f);
    elseif isstruct(a.(f)) && isstruct(d.(f)) && isscalar(a.(f)) && isscalar(d.(f))
        a.(f) = mergeInto(a.(f), d.(f));
    end
end
end