function fig = plotVoltageProfile(record, cfg, feeder)
%PLOTVOLTAGEPROFILE  Before/after voltage profile for one control decision.
%
%   FIG = PLOTVOLTAGEPROFILE(RECORD, CFG, FEEDER)
%
%   Plots bus voltage against electrical distance from the substation,
%   before and after the action, with the ANSI band shaded and device
%   locations marked. Distance rather than bus index on the x-axis is the
%   point: bus numbering on the 33-bus feeder jumps between laterals, so an
%   index plot shows discontinuities that are an artefact of numbering, not
%   of the network.

d = feeder.depth;
[~, ord] = sort(d);

fig = figure('Position',[100 100 950 560],'Color','w');
hold on;

x = [0 max(d)+1];
% HandleVisibility off on the band and the limit lines: otherwise MATLAB
% auto-names them data1/data2/data3 in the legend.
fill([x fliplr(x)], [cfg.limits.Vmin cfg.limits.Vmin cfg.limits.Vmax cfg.limits.Vmax], ...
    [0.85 0.94 0.85], 'EdgeColor','none','FaceAlpha',0.5, 'HandleVisibility','off');

yline(cfg.limits.Vmin,'r--','LineWidth',1.2,'Label','0.95 p.u.','HandleVisibility','off');
yline(cfg.limits.Vmax,'r--','LineWidth',1.2,'Label','1.05 p.u.','HandleVisibility','off');

for z = 1:max(feeder.zone)
    m = feeder.zone == z;
    [dz, oz] = sort(d(m));
    idx = find(m); idx = idx(oz);
    plot(dz, record.info.VmPre(idx),  'o--', 'LineWidth',1.0, 'MarkerSize',4, ...
        'Color',[0.6 0.6 0.6], 'HandleVisibility', ternary(z==1,'on','off'), ...
        'DisplayName','Before action');  %#ok<*AGROW>
    plot(dz, record.info.VmPost(idx), 'o-',  'LineWidth',1.6, 'MarkerSize',5, ...
        'Color',[0.10 0.35 0.75], 'HandleVisibility', ternary(z==1,'on','off'), ...
        'DisplayName','After action');
end

for k = 1:cfg.dims.nCap
    b = cfg.cap.buses(k);
    on = record.dev.capOn(k) > 0;
    plot(d(b), record.info.VmPost(b), 's', 'MarkerSize',11, 'LineWidth',1.6, ...
        'MarkerEdgeColor',[0.85 0.33 0.10], ...
        'MarkerFaceColor', ternaryVec(on,[0.95 0.6 0.3],[1 1 1]), ...
        'HandleVisibility','off');
    text(d(b), record.info.VmPost(b)+0.006, sprintf('C%d',b), ...
        'FontSize',8,'HorizontalAlignment','center');
end
for k = 1:cfg.dims.nPV
    b = cfg.pv.buses(k);
    plot(d(b), record.info.VmPost(b), '^', 'MarkerSize',10, 'LineWidth',1.4, ...
        'MarkerEdgeColor',[0.20 0.60 0.20],'MarkerFaceColor',[0.75 0.95 0.75], ...
        'HandleVisibility','off');
    text(d(b), record.info.VmPost(b)-0.010, sprintf('PV%d',b), ...
        'FontSize',8,'HorizontalAlignment','center');
end

xlabel('Electrical distance from substation (hops)');
ylabel('Bus voltage (p.u.)');
% Two-line title using the CONDENSED action: the full action text runs to
% 300+ characters when several devices move and is clipped by the axes.
if isfield(record,'actionBrief') && ~isempty(record.actionBrief)
    actTxt = record.actionBrief;
else
    [~, ~, actTxt] = describeAction(record.dev, record.devPrev, cfg);
end
title({sprintf('Voltage profile -- hour %.0f, %s scenario', record.hour, record.scenario), ...
       sprintf('Controller %s (V_{min} %.4f \\rightarrow %.4f p.u.)', ...
               actTxt, min(record.info.VmPre), min(record.info.VmPost))}, ...
    'FontSize',10,'FontWeight','normal');
legend('Location','southwest');
grid on; box on;
xlim([0 max(d)+1]);
set(gca,'FontSize',11);
hold off;
end

function o = ternary(c,a,b), if c, o=a; else, o=b; end, end
function o = ternaryVec(c,a,b), if c, o=a; else, o=b; end, end
