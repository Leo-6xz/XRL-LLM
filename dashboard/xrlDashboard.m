function app = xrlDashboard(records, explanations, cfg, feeder)
%XRLDASHBOARD  Operator-facing console for reviewing XRL-LLM decisions.
%
%   APP = XRLDASHBOARD(RECORDS, EXPLANATIONS, CFG, FEEDER)
%
%   Opens a window with, for each explained decision:
%     * the before/after voltage profile with the ANSI band and devices
%     * the SHAP attribution bar chart
%     * the layered natural-language explanation (summary / standard /
%       detailed, selectable)
%     * a what-if panel that perturbs a factor and re-queries the policy
%
%   The what-if panel is the feature operators actually use. A static
%   explanation answers "why did it do that"; the question that follows is
%   always "what would it do if conditions were different", and being able
%   to answer that in the same window is what turns the tool from a report
%   into a co-pilot.
%
%   See also GENERATELAYEREDEXPLANATIONS, PLOTVOLTAGEPROFILE.

if nargin < 3 || isempty(cfg),    cfg = xrlConfig(); end
if nargin < 4 || isempty(feeder), feeder = buildFeeder(cfg.net.dataFcn()); end
if isempty(records)
    error('xrlDashboard:noRecords','No attribution records to display.');
end
if nargin < 2, explanations = {}; end

S.records      = records;
S.explanations = explanations;
S.cfg          = cfg;
S.feeder       = feeder;
S.idx          = 1;
S.level        = 'standard';

S.fig = uifigure('Name','XRL-LLM Volt/VAR Co-Pilot  |  IEEE 33-bus', ...
    'Position',[80 80 1360 820], 'Color',[1 1 1]);

g = uigridlayout(S.fig,[3 2]);
g.RowHeight   = {40, '1x', 250};
g.ColumnWidth = {'1.35x','1x'};

% ---------------------------------------------------------------------
% Top bar: record navigation
% ---------------------------------------------------------------------
top = uigridlayout(g,[1 6]);
top.Layout.Row = 1; top.Layout.Column = [1 2];
top.ColumnWidth = {90, 90, '1x', 150, 130, 110};
top.Padding = [4 2 4 2];

S.prevBtn = uibutton(top,'Text','< Previous','ButtonPushedFcn',@(~,~)nav(-1));
S.nextBtn = uibutton(top,'Text','Next >',    'ButtonPushedFcn',@(~,~)nav(+1));
S.title   = uilabel(top,'Text','','FontWeight','bold','FontSize',13);
S.levelDD = uidropdown(top,'Items',{'summary','standard','detailed'}, ...
    'Value','standard','ValueChangedFcn',@(src,~)setLevel(src.Value));
S.jumpDD  = uidropdown(top,'Items',recordLabels(records), ...
    'ValueChangedFcn',@(src,~)jumpTo(src.Value));
S.exportBtn = uibutton(top,'Text','Export','ButtonPushedFcn',@(~,~)exportCurrent());

% ---------------------------------------------------------------------
% Left: voltage profile.  Right: SHAP attribution.
% ---------------------------------------------------------------------
S.axV = uiaxes(g); S.axV.Layout.Row = 2; S.axV.Layout.Column = 1;
S.axS = uiaxes(g); S.axS.Layout.Row = 2; S.axS.Layout.Column = 2;

% ---------------------------------------------------------------------
% Bottom left: explanation text
% ---------------------------------------------------------------------
explPanel = uipanel(g,'Title','Natural-language justification');
explPanel.Layout.Row = 3; explPanel.Layout.Column = 1;
eg = uigridlayout(explPanel,[1 1]); eg.Padding = [6 6 6 6];
S.explBox = uitextarea(eg,'Editable','off','FontSize',13,'WordWrap','on');

% ---------------------------------------------------------------------
% Bottom right: what-if
% ---------------------------------------------------------------------
wPanel = uipanel(g,'Title','What-if analysis');
wPanel.Layout.Row = 3; wPanel.Layout.Column = 2;
wg = uigridlayout(wPanel,[4 2]);
wg.RowHeight = {28,28,32,'1x'}; wg.ColumnWidth = {130,'1x'};

uilabel(wg,'Text','Perturb factor:');
S.wfFactor = uidropdown(wg,'Items',records(1).playerNames);

uilabel(wg,'Text','Change (%):');
S.wfPct = uislider(wg,'Limits',[-30 30],'Value',10, ...
    'MajorTicks',-30:15:30);

S.wfBtn = uibutton(wg,'Text','Re-query the policy', ...
    'ButtonPushedFcn',@(~,~)whatIf());
S.wfBtn.Layout.Row = 3; S.wfBtn.Layout.Column = [1 2];

S.wfOut = uitextarea(wg,'Editable','off','FontSize',12,'WordWrap','on');
S.wfOut.Layout.Row = 4; S.wfOut.Layout.Column = [1 2];

guidata(S.fig, S);
refresh();
app = S.fig;

% =====================================================================
% Callbacks (nested so they share S through GUIDATA)
% =====================================================================
    function nav(delta)
        S = guidata(S.fig);
        S.idx = max(1, min(numel(S.records), S.idx + delta));
        guidata(S.fig, S);
        refresh();
    end

    function jumpTo(label)
        S = guidata(S.fig);
        labels = recordLabels(S.records);
        k = find(strcmp(labels, label), 1);
        if ~isempty(k), S.idx = k; end
        guidata(S.fig, S);
        refresh();
    end

    function setLevel(L)
        S = guidata(S.fig);
        S.level = L;
        guidata(S.fig, S);
        refresh();
    end

    function refresh()
        S = guidata(S.fig);
        r = S.records(S.idx);

        S.title.Text = sprintf('Decision %d of %d  |  episode %d, hour %.0f  |  %s scenario', ...
            S.idx, numel(S.records), r.episode, r.hour, r.scenario);
        S.jumpDD.Value = S.jumpDD.Items{S.idx};

        drawVoltage(S.axV, r, S.cfg, S.feeder);
        drawSHAP(S.axS, r);

        S.explBox.Value = explanationFor(S, r);
        guidata(S.fig, S);
    end

    function txt = explanationFor(S, r)
        k = S.idx;
        if ~isempty(S.explanations) && k <= numel(S.explanations)
            e = S.explanations{k};
            if isstruct(e) && isfield(e, S.level)
                txt = splitLines(e.(S.level));
                return
            elseif ischar(e) || isstring(e)
                txt = splitLines(char(e));
                return
            end
        end
        % Nothing precomputed: generate offline so the panel is never blank.
        txt = splitLines(generateTemplateExplanation(r, S.cfg, S.feeder, S.level));
    end

    function whatIf()
        S = guidata(S.fig);
        r = S.records(S.idx);

        g_ = find(strcmp(r.playerNames, S.wfFactor.Value), 1);
        if isempty(g_)
            S.wfOut.Value = {'Unknown factor.'}; return
        end

        map = featureMapping(S.cfg);
        switch lower(S.cfg.shap.mode)
            case 'grouped', groups = map.groupMembers;
            otherwise,      groups = num2cell(1:S.cfg.dims.nObs);
        end
        idxF = groups{g_};

        pct  = S.wfPct.Value / 100;
        obs0 = r.obs(:)';
        obsP = obs0;
        obsP(idxF) = obsP(idxF) .* (1 + pct);

        pf = makePolicyFcn('rule', S.cfg);   % offline-safe default
        if isfield(S,'policyFcn') && ~isempty(S.policyFcn)
            pf = S.policyFcn;
        end

        a0 = pf(obs0); a1 = pf(obsP);
        d  = a1 - a0;

        names = map.actionNames;
        [~, big] = max(abs(d));

        lines = { ...
            sprintf('Perturbing "%s" by %+.0f%%:', S.wfFactor.Value, 100*pct), ...
            sprintf('  Largest response: %s, %+.4f (normalised).', names{big}, d(big)), ...
            sprintf('  Total action change (L1): %.4f.', sum(abs(d)))};

        if abs(d(big)) < 1e-4
            lines{end+1} = '  The policy is insensitive to this factor here.';
        end
        if g_ == r.topFeatures(1).idx
            lines{end+1} = sprintf(['  This is the factor SHAP ranked first (%.0f%% of ' ...
                'attribution), so a large response is the expected, faithful behaviour.'], ...
                100*r.topFeatures(1).share);
        end

        S.wfOut.Value = lines;
        guidata(S.fig, S);
    end

    function exportCurrent()
        S = guidata(S.fig);
        r = S.records(S.idx);
        f = fullfile(S.cfg.io.resultsDir, sprintf('decision_%03d.txt', S.idx));
        if ~exist(S.cfg.io.resultsDir,'dir'), mkdir(S.cfg.io.resultsDir); end
        fid = fopen(f,'w');
        fprintf(fid, 'Episode %d, hour %.0f, %s scenario\n\n', r.episode, r.hour, r.scenario);
        fprintf(fid, 'ACTION: %s\n\n', r.actionText);
        fprintf(fid, 'ATTRIBUTION:\n');
        for i = 1:numel(r.topFeatures)
            fprintf(fid, '  %-40s %+.4f (%.0f%%)\n', r.topFeatures(i).name, ...
                r.topFeatures(i).value, 100*r.topFeatures(i).share);
        end
        fprintf(fid, '\nEXPLANATION:\n%s\n', strjoin(S.explBox.Value, newline));
        fclose(fid);
        uialert(S.fig, sprintf('Written to %s', f), 'Export complete', 'Icon','success');
    end
end

% =====================================================================
function drawVoltage(ax, r, cfg, feeder)
cla(ax); hold(ax,'on');
d = feeder.depth;
x = [0 max(d)+1];
fill(ax, [x fliplr(x)], ...
    [cfg.limits.Vmin cfg.limits.Vmin cfg.limits.Vmax cfg.limits.Vmax], ...
    [0.86 0.94 0.86], 'EdgeColor','none','FaceAlpha',0.6);
for z = 1:max(feeder.zone)
    m = feeder.zone == z;
    [dz, oz] = sort(d(m)); idx = find(m); idx = idx(oz);
    plot(ax, dz, r.info.VmPre(idx),  'o--','Color',[0.65 0.65 0.65],'LineWidth',1.0,'MarkerSize',4);
    plot(ax, dz, r.info.VmPost(idx), 'o-', 'Color',[0.10 0.35 0.75],'LineWidth',1.6,'MarkerSize',5);
end
yline(ax, cfg.limits.Vmin,'r--'); yline(ax, cfg.limits.Vmax,'r--');
for k = 1:cfg.dims.nCap
    b = cfg.cap.buses(k);
    on = r.dev.capOn(k) > 0;
    fc = [1 1 1]; if on, fc = [0.95 0.60 0.30]; end
    plot(ax, d(b), r.info.VmPost(b), 's','MarkerSize',10,'LineWidth',1.5, ...
        'MarkerEdgeColor',[0.85 0.33 0.10],'MarkerFaceColor',fc);
end
for k = 1:cfg.dims.nPV
    b = cfg.pv.buses(k);
    plot(ax, d(b), r.info.VmPost(b), '^','MarkerSize',9,'LineWidth',1.3, ...
        'MarkerEdgeColor',[0.20 0.60 0.20],'MarkerFaceColor',[0.75 0.95 0.75]);
end
hold(ax,'off');
xlabel(ax,'Distance from substation (hops)');
ylabel(ax,'Voltage (p.u.)');
title(ax, sprintf('Voltage profile  (grey = before, blue = after)  |  min %.4f  max %.4f', ...
    r.info.Vmin, r.info.Vmax));
grid(ax,'on'); xlim(ax,[0 max(d)+1]);
end

% =====================================================================
function drawSHAP(ax, r)
cla(ax);
phi = r.phi(:);
[~, o] = sort(abs(phi),'descend');
k = min(8, numel(phi));
o = o(1:k);
v = phi(o);
b = barh(ax, v, 'FaceColor','flat');
for i = 1:k
    if v(i) >= 0
        b.CData(i,:) = [0.20 0.60 0.90];
    else
        b.CData(i,:) = [0.90 0.30 0.30];
    end
end
set(ax,'YTick',1:k,'YTickLabel',r.playerNames(o),'YDir','reverse');
xlabel(ax,'SHAP value');
title(ax, sprintf('Attribution for: %s', r.shapMeta.explainedChannelName), ...
    'Interpreter','none');
grid(ax,'on');
xline(ax, 0, 'k-');
end

% =====================================================================
function labels = recordLabels(records)
labels = arrayfun(@(r) sprintf('ep%d t%02d (%s)', r.episode, r.t, r.scenario), ...
    records, 'UniformOutput', false);
end

function c = splitLines(s)
c = strsplit(char(s), newline);
c = c(:);
end
