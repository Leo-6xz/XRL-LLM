function [groups, names, tag] = shapGroups(cfg, map)
%SHAPGROUPS  Resolve the active SHAP player set from the config.
%
%   [GROUPS, NAMES, TAG] = SHAPGROUPS(CFG, MAP)
%
%   The single place that decides fine vs coarse vs raw. Both
%   COLLECTATTRIBUTIONS and COUNTERFACTUALTEST must call this, or they can
%   disagree about how many players there are.
if nargin < 2 || isempty(map), map = featureMapping(cfg); end

switch lower(cfg.shap.mode)
    case 'grouped'
        grouping = 'fine';
        if isfield(cfg.shap,'grouping'), grouping = cfg.shap.grouping; end
        if strcmpi(grouping,'coarse')
            groups = map.coarse.groupMembers;
            names  = map.coarse.groupNames;
            tag    = 'coarse';
        else
            groups = map.groupMembers;
            names  = map.groupNames;
            tag    = 'fine';
        end
    case 'raw'
        groups = num2cell(1:cfg.dims.nObs);
        names  = map.labels;
        tag    = 'raw';
    otherwise
        error('shapGroups:badMode', 'cfg.shap.mode must be grouped or raw.');
end
end