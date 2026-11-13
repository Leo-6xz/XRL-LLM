%% make_fig9_dashboard.m
% Fig. 9: operator-facing XAI dashboard for ONE explained decision, as a
% static, print-ready figure (the interactive app is dashboard/xrlDashboard).
%
% Every number on the figure is read from the saved Phase 2 results or
% recomputed from them; nothing is typed in by hand. 
%
% Panels
%   Header  time, regime, net substation flow, feeder status before acting
%   (a)     what the controller did: every moved device, before -> after,
%           and the outcome (V_max, V_min, violated buses, loss)
%   (b)     voltage profile before/after against electrical distance
%   (c)     why: grouped Shapley attribution of the explained setpoint
%   (d)     the natural-language justification (summary + standard level)
%   (e)     can it be trusted: Shapley efficiency residual, counterfactual
%           check of the leading factor, sensitivity-predicted vs realised
%           voltage change, text/attribution consistency
%   (f)     what conventional VVC would have done in the same state
%
% Requires: results\xrl_records.mat (Phase 2). Optional: the trained agent
% (for the per-decision counterfactual check) and Ollama (LIVE_LLM = true,
% to regenerate all three levels and time them).

% Works from the project folder or from the folder that contains it.
if ~exist(fullfile(pwd,'setupPaths.m'),'file') && exist('XRL_LLM_Project','dir')
    cd XRL_LLM_Project;
end
setupPaths;

PICK     = 4;       % record index; 4 = the decision shown in Fig. 4
LIVE_LLM = false;   % true: regenerate the three levels now and time them
OUTNAME  = 'fig9_xai_dashboard';

% ------------------------------------------------------------------ load
S      = load(fullfile('results','xrl_records.mat'));
recs   = S.records;
cfg    = mergeConfigDefaults(S.cfg);
feeder = buildFeeder(cfg.net.dataFcn());
r      = recs(PICK);
base   = cfg.net.baseMVA;

% Explanation text: live > saved three-level set > saved standard + template
expl = struct('summary','', 'standard','', 'detailed','');
src  = '';
llmLatency = NaN; consistent = NaN;
if LIVE_LLM
    llm = connectLLM(cfg);
    [e, m] = generateLayeredExplanations(r, llm, cfg, feeder);
    expl = e; llmLatency = m.totalLatencySec; consistent = m.consistent;
    src = sprintf('%s, generated live', cfg.llm.model);
else
    f = dir(fullfile('results','phase_explain_*.mat'));
    if ~isempty(f)
        [~, iNew] = max([f.datenum]);
        try
            P = load(fullfile(f(iNew).folder, f(iNew).name), 'out');
            e = P.out.explanations{PICK};
            if isstruct(e), expl = e; src = sprintf('%s, saved Phase 2 output', cfg.llm.model); end
        catch
        end
    end
    if isempty(src)
        expl.standard = S.explanations{PICK};
        expl.summary  = generateTemplateExplanation(r, cfg, feeder, 'summary');
        src = sprintf('standard: %s (saved); summary: template', cfg.llm.model);
    end
end

% ---------------------------------------------------------- derived data
VmPre  = r.info.VmPre(:);  VmPost = r.info.VmPost(:);
lim    = cfg.limits;
nViol  = @(v) nnz(v < lim.Vmin | v > lim.Vmax);
lossPre  = r.state.pf.Ploss * base * 1000;
lossPost = r.info.PlossKW;
pvMW   = sum(r.state.pvP) * base;
loadMW = sum(r.state.Pload) * base;

dev  = r.dev;  prev = r.devPrev;
if isempty(prev)
    prev = struct('tap',0,'capOn',zeros(size(dev.capOn)),'invQ',zeros(size(dev.invQ)));
end

% Critical bus: the side of the band under more pressure before acting
if max(VmPre) - lim.Vnom >= lim.Vnom - min(VmPre)
    [~, crit] = max(VmPre); critTxt = 'highest-voltage bus';
else
    [~, crit] = min(VmPre); critTxt = 'lowest-voltage bus';
end

% Sensitivity-predicted voltage change of the whole action (linearised)
Ssen = r.sensitivity;
dVpred = Ssen.dVdT * (dev.tap - prev.tap);
bd = r.info.interpBreakdown;
for k = find(bd.moved(:))'
    if k > 1, dVpred = dVpred + Ssen.dVdQ(:, k-1) * bd.dQ(k); end
end
dVreal = VmPost - VmPre;

% Conventional VVC in the same state
ruleA   = rulePolicy(r.obs(:)', cfg, feeder);
devRule = decodeAction(ruleA(:), cfg, r.state.pvP);
pfRule  = solveWithDevices(r.state, devRule, cfg, feeder);
[~, ~, ruleBrief] = describeAction(devRule, r.devPrev, cfg);
ruleBrief = regexprep(ruleBrief, '\<1 ([a-zA-Z ]+?)\(s\)', '1 $1');
ruleBrief = regexprep(ruleBrief, '(\w+)\(s\)', '$1s');

% Per-decision counterfactual check (needs the trained agent)
cf = struct('rho',NaN,'top1',NaN,'sign',NaN);
try
    A  = load(cfg.train.savePath, 'agent');
    pf = makePolicyFcn(A.agent, cfg);
    % The probe must use the SAME background as the attribution. With a
    % single record and no background, counterfactualTest would substitute
    % the record with itself and measure nothing. Re-derive the Phase 2
    % background (same seed, same 12 episodes; only one record explained).
    [~, bgMeta] = collectAttributions(pf, cfg, feeder, 12, ...
        struct('maxRecords', 1, 'verbose', false));
    q  = counterfactualTest(r, pf, cfg, feeder, ...
        struct('background', bgMeta.background, 'verbose', false));
    cf.rho = q.rankCorr(1); cf.top1 = q.top1Hit(1); cf.sign = q.signOK(1);
catch ME
    warning('fig9:noCF', 'Counterfactual check skipped: %s', ME.message);
end
if isnan(consistent)
    key = lower(strtok(r.topFeatures(1).name));
    consistent = contains(lower(expl.standard), key);
end
% Does the text state the EXPECTED voltage change in the direction the physics
% gives? Only effect clauses are read, not the trigger ("voltage drop").
% (an LLM can name the right factor and still reverse the effect)
txtAll = lower([expl.summary ' ' expl.standard]);
upPat  = '(expected to|will|would|should)\s+(\w+\s+){0,1}(increase|raise|boost|lift)\w*\s+(\w+\s+){0,4}voltage';
dnPat  = '(expected to|will|would|should)\s+(\w+\s+){0,1}(decrease|reduce|lower|bring down)\w*\s+(\w+\s+){0,4}voltage';
saysUp = ~isempty(regexp(txtAll, upPat, 'once'));
saysDn = ~isempty(regexp(txtAll, dnPat, 'once'));
realDir = sign(dVreal(crit));
if saysUp == saysDn
    dirOK = NaN;                         % no (or ambiguous) direction stated
else
    dirOK = (saysUp && realDir > 0) || (saysDn && realDir < 0);
end

% ------------------------------------------------------------- styling
C.blue  = [0.10 0.35 0.75];  C.grey = [0.60 0.60 0.60];  C.red = [0.80 0.20 0.15];
C.green = [0.15 0.55 0.25];  C.amber = [0.90 0.60 0.00]; C.band = [0.88 0.95 0.88];
C.frame = [0.80 0.80 0.80];  C.ink  = [0.15 0.15 0.15];
fs = 7;                                    % body font size (pt)
fig = figure('Units','inches','Position',[0.5 0.5 7.16 6.3],'Color','w');

% ---------------------------------------------------------------- header
regimeName = containers.Map( ...
    {'reversePower','highPV','heavyLoad','lightLoad','nominal'}, ...
    {'reverse-power day','high-PV day','heavy-load day','light-load day','nominal day'});
rg = r.scenario; if isKey(regimeName, rg), rg = regimeName(rg); end
if nViol(VmPre) > 0
    stTxt = sprintf('VIOLATION (%d buses)', nViol(VmPre)); stCol = C.red;
elseif max(VmPre) > lim.VsoftMax || min(VmPre) < lim.VsoftMin
    stTxt = 'APPROACHING LIMIT'; stCol = C.amber;
else
    stTxt = 'NORMAL'; stCol = C.green;
end
flowTxt = 'forward flow';
if pvMW > loadMW, flowTxt = 'reverse flow at substation'; end
axH = axes(fig,'Position',[0.01 0.925 0.98 0.07]); axis(axH,'off');
rectangle(axH,'Position',[0 0 1 1],'FaceColor',[0.95 0.96 0.98],'EdgeColor',C.frame);
text(axH,0.012,0.68,sprintf('Decision %02d:00, %s', round(r.hour), rg), ...
    'FontSize',fs+2,'FontWeight','bold','Color',C.ink);
text(axH,0.012,0.25,sprintf('PV %.2f MW | load %.2f MW | %s | V_{max} %.4f p.u., V_{min} %.4f p.u. before acting', ...
    pvMW, loadMW, flowTxt, max(VmPre), min(VmPre)),'FontSize',fs,'Color',C.ink);
rectangle(axH,'Position',[0.80 0.18 0.19 0.64],'Curvature',0.3,'FaceColor',stCol,'EdgeColor','none');
text(axH,0.895,0.50,stTxt,'FontSize',fs,'FontWeight','bold','Color','w', ...
    'HorizontalAlignment','center');
xlim(axH,[0 1]); ylim(axH,[0 1]);

% ------------------------------------------------- (a) action and outcome
axA = panel(fig,[0.01 0.62 0.47 0.29],'(a) Action taken and outcome',C,fs);
rows = cell(0,4);
if dev.tap ~= prev.tap
    rows(end+1,:) = {'OLTC tap', sprintf('%+d', prev.tap), sprintf('%+d', dev.tap), ...
        sprintf('V_0 %.4f p.u.', dev.V0)};
end
for k = 1:numel(dev.capOn)
    if dev.capOn(k) ~= prev.capOn(k)
        st = {'open','closed'};
        rows(end+1,:) = {sprintf('Capacitor, bus %d', cfg.cap.buses(k)), st{1+(prev.capOn(k)>0)}, ...
            st{1+(dev.capOn(k)>0)}, sprintf('%+.0f kvar', (dev.capOn(k)-prev.capOn(k))*cfg.cap.kvar(k))}; %#ok<SAGROW>
    end
end
for k = 1:numel(dev.invQ)
    if abs(dev.invQ(k) - prev.invQ(k)) * base > 5e-4
        rows(end+1,:) = {sprintf('PV inverter, bus %d', cfg.pv.buses(k)), ...
            sprintf('%+.3f', prev.invQ(k)*base), sprintf('%+.3f', dev.invQ(k)*base), 'Mvar'}; %#ok<SAGROW>
    end
end
nUnch = (1 + numel(dev.capOn) + numel(dev.invQ)) - size(rows,1);
y = 0.80;
hdr = {'Device','Before','After','Change'}; xc = [0.03 0.47 0.63 0.79];
for j = 1:4, text(axA,xc(j),y,hdr{j},'FontSize',fs,'FontWeight','bold','Color',C.ink); end
for i = 1:size(rows,1)
    y = y - 0.088;
    for j = 1:4, text(axA,xc(j),y,rows{i,j},'FontSize',fs,'Color',C.ink); end
end
y = y - 0.088;
text(axA,0.03,y,sprintf('%d other devices held', nUnch),'FontSize',fs,'Color',C.grey,'FontAngle','italic');
y = y - 0.105;
line(axA,[0.03 0.97],[y+0.06 y+0.06],'Color',C.frame);
out = {sprintf('V_{max} %.4f \\rightarrow %.4f p.u.', max(VmPre), max(VmPost)), ...
       sprintf('V_{min} %.4f \\rightarrow %.4f p.u.', min(VmPre), min(VmPost)), ...
       sprintf('Buses outside band %d \\rightarrow %d', nViol(VmPre), nViol(VmPost)), ...
       sprintf('Active loss %.1f \\rightarrow %.1f kW', lossPre, lossPost)};
text(axA,0.03,y,out{1},'FontSize',fs); text(axA,0.52,y,out{2},'FontSize',fs);
text(axA,0.03,y-0.09,out{3},'FontSize',fs); text(axA,0.52,y-0.09,out{4},'FontSize',fs);

% ------------------------------------------------------- (b) voltage profile
axB = axes(fig,'Position',[0.075 0.085 0.40 0.45]); hold(axB,'on');
d = feeder.depth(:); xm = max(d) + 1;
fill(axB,[0 xm xm 0],[lim.Vmin lim.Vmin lim.Vmax lim.Vmax],C.band,'EdgeColor','none','HandleVisibility','off');
plot(axB,[0 xm],[lim.VsoftMax lim.VsoftMax],':','Color',C.amber,'HandleVisibility','off');
plot(axB,[0 xm],[lim.VsoftMin lim.VsoftMin],':','Color',C.amber,'HandleVisibility','off');
for z = 1:max(feeder.zone)
    m = find(feeder.zone == z); [~, o] = sort(d(m)); m = m(o);
    if z > 1, m = [feeder.parentBus(m(1)); m]; end          % join lateral to its root
    plot(axB,d(m),VmPre(m),'--','Color',C.grey,'LineWidth',0.8,'HandleVisibility',onoff(z==1),'DisplayName','Before');
    plot(axB,d(m),VmPost(m),'-','Color',C.blue,'LineWidth',1.2,'HandleVisibility',onoff(z==1),'DisplayName','After');
end
for k = 1:numel(cfg.cap.buses)
    b = cfg.cap.buses(k);
    plot(axB,d(b),VmPost(b),'s','MarkerSize',5,'MarkerEdgeColor',[0.85 0.33 0.10], ...
        'MarkerFaceColor',pick(dev.capOn(k)>0,[0.95 0.60 0.30],[1 1 1]),'HandleVisibility','off');
end
for k = 1:numel(cfg.pv.buses)
    b = cfg.pv.buses(k);
    plot(axB,d(b),VmPost(b),'^','MarkerSize',5,'MarkerEdgeColor',C.green,'MarkerFaceColor',[0.75 0.95 0.75],'HandleVisibility','off');
end
plot(axB,d(crit),VmPost(crit),'o','MarkerSize',7,'MarkerEdgeColor',C.red,'LineWidth',1,'HandleVisibility','off');
xlim(axB,[0 xm]);
ylo = min([VmPre; VmPost; lim.Vmin]) - 0.01; yhi = max([VmPre; VmPost; lim.Vmax]) + 0.01;
ylim(axB,[ylo yhi]); grid(axB,'on'); box(axB,'on');
set(axB,'FontSize',fs);
xlabel(axB,'Electrical distance from substation (hops)','FontSize',fs);
ylabel(axB,'Bus voltage (p.u.)','FontSize',fs);
legend(axB,'Location','northwest','FontSize',fs-1,'Box','off');
title(axB,'(b) Voltage profile','FontSize',fs+1,'FontWeight','bold','HorizontalAlignment','left', ...
    'Units','normalized','Position',[0 1.01 0]);

% ---------------------------------------------------------- (c) attribution
axC = axes(fig,'Position',[0.735 0.635 0.245 0.245]);
phi = r.phi(:); names = r.playerNames(:);
[~, o] = sort(abs(phi),'descend'); o = o(1:min(6,numel(o)));
v = phi(o); share = abs(v) / sum(abs(phi));
bh = barh(axC, flipud(v), 0.7, 'FaceColor','flat','EdgeColor','none');
cols = repmat(C.blue, numel(v), 1); cols(flipud(v) < 0, :) = repmat(C.red, nnz(v < 0), 1);
bh.CData = cols;
lbl = cellfun(@(s) shortName(s), flipud(names(o)), 'UniformOutput', false);
set(axC,'YTick',1:numel(v),'YTickLabel',lbl,'FontSize',fs-0.5,'TickLabelInterpreter','none');
xl = max(abs(v)) * 1.45; xlim(axC,[-xl xl]); grid(axC,'on'); box(axC,'off');
fv = flipud(share);
for i = 1:numel(v)
    text(axC, flipud_v(v,i) + sign(flipud_v(v,i))*0.03*xl, i, sprintf('%.0f%%', 100*fv(i)), ...
        'FontSize',fs-1,'HorizontalAlignment',pick(flipud_v(v,i)>=0,'left','right'),'VerticalAlignment','middle');
end
xlabel(axC, sprintf('Shapley value on: %s', r.shapMeta.explainedChannelName), 'FontSize',fs-0.5,'Interpreter','none');
annotation(fig,'textbox',[0.515 0.885 0.47 0.03],'String','(c) Why: grouped Shapley attribution', ...
    'FontSize',fs+1,'FontWeight','bold','EdgeColor','none','Margin',0);

% ------------------------------------------------------ (d) justification
axD = panel(fig,[0.515 0.30 0.475 0.285],'(d) Justification shown to the operator',C,fs);
txt = { ['SUMMARY. ' strtrim(expl.summary)], '', ['STANDARD. ' strtrim(expl.standard)] };
txt = wrapLines(txt, 72);
maxL = 13;
if numel(txt) > maxL, txt = [txt(1:maxL-1); {'[...] detailed level available on request'}]; end
text(axD,0.025,0.86,txt,'FontSize',fs-0.5,'VerticalAlignment','top','Interpreter','none','Color',C.ink);
text(axD,0.975,0.035,['Source: ' src],'FontSize',fs-1.5,'Color',C.grey,'HorizontalAlignment','right','Interpreter','none');

% --------------------------------------------------------- (e) trust checks
axE = panel(fig,[0.515 0.015 0.235 0.265],'(e) Can it be trusted?',C,fs);
chk = { ...
  'Efficiency residual', sprintf('%.1e', r.shapMeta.efficiencyGap), r.shapMeta.efficiencyGap < 1e-6; ...
  'Leading factor sign', yn(cf.sign), cf.sign; ...
  'Leading factor rank', yn(cf.top1), cf.top1; ...
  sprintf('\\DeltaV_{%d} pred./real', crit), sprintf('%+.4f/%+.4f', dVpred(crit), dVreal(crit)), ...
      sign(dVpred(crit)) == sign(dVreal(crit)) && abs(dVpred(crit)-dVreal(crit)) <= 0.30*abs(dVreal(crit)); ...
  'Text cites top factor', yn(consistent), consistent; ...
  'Text direction = physics', yn(dirOK), dirOK};
if ~isnan(llmLatency)
    chk(end+1,:) = {'Generation latency', sprintf('%.1f s', llmLatency), llmLatency <= cfg.eval.targetLatencySec};
end
y = 0.80;
for i = 1:size(chk,1)
    ok = chk{i,3};
    if isnan(ok), mk = '-'; mc = C.grey; elseif ok, mk = '\surd'; mc = C.green; else, mk = '\times'; mc = C.red; end
    text(axE,0.04,y,mk,'FontSize',fs+1,'Color',mc,'FontWeight','bold');
    text(axE,0.14,y,chk{i,1},'FontSize',fs-0.5);
    text(axE,0.97,y,chk{i,2},'FontSize',fs-1,'HorizontalAlignment','right');
    y = y - 0.115;
end
text(axE,0.04,0.04,sprintf('Critical point: %s', critTxt),'FontSize',fs-1.5,'Color',C.grey);

% ---------------------------------------------------- (f) conventional VVC
axF = panel(fig,[0.755 0.015 0.235 0.265],'(f) Conventional VVC',C,fs);
ft = wrapLines({['Would have: ' ruleBrief]}, 36);
text(axF,0.04,0.86,{'Same state, same instant:'},'FontSize',fs-1,'Color',C.grey);
text(axF,0.04,0.76,ft,'FontSize',fs-0.5,'VerticalAlignment','top','Interpreter','none');
yb = 0.76 - 0.105*numel(ft) - 0.04;
text(axF,0.04,yb,      sprintf('V_{max} %.4f p.u.  (XRL %.4f)', max(pfRule.Vm), max(VmPost)),'FontSize',fs-0.5);
text(axF,0.04,yb-0.11, sprintf('V_{min} %.4f p.u.  (XRL %.4f)', min(pfRule.Vm), min(VmPost)),'FontSize',fs-0.5);
text(axF,0.04,yb-0.22, sprintf('Loss %.1f kW  (XRL %.1f kW)', pfRule.Ploss*base*1000, lossPost),'FontSize',fs-0.5);
text(axF,0.04,yb-0.33, sprintf('Outside band: %d  (XRL %d)', nViol(pfRule.Vm), nViol(VmPost)),'FontSize',fs-0.5);

% ------------------------------------------------------------------ export
exportgraphics(fig, fullfile('results',[OUTNAME '.png']), 'Resolution', 600);
exportgraphics(fig, fullfile('results',[OUTNAME '.pdf']), 'ContentType', 'vector');
fprintf('Saved results\\%s.png and .pdf (record %d)\n', OUTNAME, PICK);
fprintf('Explanation source: %s\n', src);

% ========================================================== local helpers
function ax = panel(fig, pos, ttl, C, fs)
ax = axes(fig,'Position',pos); axis(ax,'off'); xlim(ax,[0 1]); ylim(ax,[0 1]);
rectangle(ax,'Position',[0 0 1 1],'EdgeColor',C.frame,'LineWidth',0.6);
text(ax,0.02,0.955,ttl,'FontSize',fs+1,'FontWeight','bold','VerticalAlignment','top');
end
function s = onoff(c), if c, s = 'on'; else, s = 'off'; end, end
function o = pick(c, a, b), if c, o = a; else, o = b; end, end
function s = yn(v)
if isnan(v), s = 'n/a'; elseif v, s = 'pass'; else, s = 'fail'; end
end
function x = flipud_v(v, i), w = flipud(v(:)); x = w(i); end
function s = shortName(s)
s = regexprep(s, ' voltages', ' V');
s = regexprep(s, 'Previous control action', 'Previous action');
s = strrep(s, 'Main-feeder load (bus 2-18)', 'Main-feeder load (bus 2-17)');   % label fix, indices 34:49
s = strrep(s, 'Lateral load (bus 19-33)', 'End and lateral load (bus 18-33)');  % indices 50:65
end
function out = wrapLines(paras, width)
out = {};
for p = 1:numel(paras)
    words = strsplit(paras{p}, ' ');
    if isempty(paras{p}), out{end+1,1} = ''; continue; end %#ok<AGROW>
    line = '';
    for w = 1:numel(words)
        if isempty(line), cand = words{w}; else, cand = [line ' ' words{w}]; end
        if numel(cand) > width && ~isempty(line)
            out{end+1,1} = line; line = words{w}; %#ok<AGROW>
        else
            line = cand;
        end
    end
    out{end+1,1} = line; %#ok<AGROW>
end
end
