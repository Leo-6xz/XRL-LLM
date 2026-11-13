function fig = plotSHAPBar(phiMatrix, playerNames, titleStr, xlabelStr)
%PLOTSHAPBAR  Global SHAP importance bar chart (signed mean).
%
%   FIG = PLOTSHAPBAR(PHIMATRIX, PLAYERNAMES, TITLESTR, XLABELSTR)
%
%   PHIMATRIX is [nRecords x M]. Bars show the SIGNED mean attribution,
%   sorted by absolute magnitude, blue for positive and red for negative --
%   the layout supplied in the project brief, adapted to the VVC feature set.

if nargin < 3 || isempty(titleStr),  titleStr  = 'Global SHAP feature importance'; end
if nargin < 4 || isempty(xlabelStr), xlabelStr = 'Mean SHAP value (impact on normalised setpoint)'; end

meanSigned = mean(phiMatrix, 1);
[~, idx]   = sort(abs(meanSigned), 'descend');
sortedMean = meanSigned(idx);
labels     = playerNames(idx);

fig = figure('Position', [100 100 900 620], 'Color', 'w');
hBar = barh(sortedMean, 'FaceColor', 'flat');
for k = 1:numel(sortedMean)
    if sortedMean(k) >= 0
        hBar.CData(k,:) = [0.20 0.60 0.90];
    else
        hBar.CData(k,:) = [0.90 0.30 0.30];
    end
end

set(gca, 'YTick', 1:numel(labels), 'YTickLabel', labels, 'YDir', 'reverse');
xlabel(xlabelStr);
title(titleStr, 'Interpreter', 'none');
grid on; box on;
set(gca, 'FontSize', 11, 'LineWidth', 1);

span = max(abs(sortedMean));
if span == 0, span = 1; end
for k = 1:numel(sortedMean)
    val = sortedMean(k);
    if val >= 0
        text(val + span*0.02, k, sprintf('%+.4f', val), ...
            'HorizontalAlignment','left','VerticalAlignment','middle','FontSize',9);
    else
        text(val - span*0.02, k, sprintf('%.4f', val), ...
            'HorizontalAlignment','right','VerticalAlignment','middle','FontSize',9);
    end
end
xlim([min(0, min(sortedMean)) - span*0.25, max(0, max(sortedMean)) + span*0.25]);
end
