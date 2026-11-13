function V = summariseGroupValues(records, groups)
%SUMMARISEGROUPVALUES  Representative value per SHAP group, per record.
%
%   V = SUMMARISEGROUPVALUES(RECORDS, GROUPS) returns [nRecords x M], the
%   mean of the observation features belonging to each group. This is the
%   colour axis for PLOTSHAPBEESWARM when players are groups rather than
%   individual features.
%
%   The mean is the right summary here because every group in
%   FEATUREMAPPING is homogeneous in units (all voltages, or all loads),
%   so averaging is meaningful. If you add a heterogeneous group, replace
%   the mean with a z-scored mean or the beeswarm colours become
%   uninterpretable.

n = numel(records);
M = numel(groups);
V = zeros(n, M);
for i = 1:n
    o = records(i).obs(:);
    for g = 1:M
        V(i,g) = mean(o(groups{g}));
    end
end
end
