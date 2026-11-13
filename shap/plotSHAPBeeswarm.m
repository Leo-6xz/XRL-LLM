function fig = plotSHAPBeeswarm(phiMatrix, featureValues, playerNames, titleStr)
%PLOTSHAPBEESWARM  SHAP summary (beeswarm) plot.
%
%   FIG = PLOTSHAPBEESWARM(PHIMATRIX, FEATUREVALUES, PLAYERNAMES, TITLESTR)
%
%   PHIMATRIX     [nRecords x M] attributions
%   FEATUREVALUES [nRecords x M] the representative value of each player
%                 in each record, used for the colour axis. For GROUPED
%                 SHAP, pass the group MEAN of the underlying features --
%                 SUMMARISEGROUPVALUES does this.
%
%   Each point is one explained decision. Horizontal position is the
%   attribution; colour is the (normalized) feature value. The classic
%   reading: a clean blue-to-red left-to-right gradient means a monotone
%   relationship, whereas colour mixing at the same x means the feature
%   only matters in interaction with something else.

if nargin < 4 || isempty(titleStr), titleStr = 'SHAP summary plot'; end

meanSigned = mean(phiMatrix, 1);
[~, idx]   = sort(abs(meanSigned), 'descend');
sortedSHAP = phiMatrix(:, idx);
sortedVals = featureValues(:, idx);
labels     = playerNames(idx);
nFeat      = size(sortedSHAP, 2);

fig = figure('Position', [100 100 950 640], 'Color', 'w');
hold on;

for j = 1:nFeat
    s = sortedSHAP(:, j);
    v = sortedVals(:, j);

    vmin = min(v); vmax = max(v);
    if vmax - vmin == 0
        c = 0.5 * ones(size(v));
    else
        c = (v - vmin) / (vmax - vmin);
    end

    jitter = (rand(size(s)) - 0.5) * 0.6;
    scatter(s, j + jitter, 32, c, 'filled', 'MarkerFaceAlpha', 0.65);
end

colormap(jet);
cb = colorbar;
cb.Label.String = 'Feature value (low \rightarrow high)';

set(gca, 'YTick', 1:nFeat, 'YTickLabel', labels, 'YDir', 'reverse');
xlabel('SHAP value (impact on normalised setpoint)');
ylabel('Feature group');
title(titleStr, 'Interpreter', 'none');
grid on; box on;
xline(0, '--k', 'LineWidth', 1);
ylim([0.5, nFeat + 0.5]);
set(gca, 'FontSize', 11, 'LineWidth', 1);
hold off;
end
