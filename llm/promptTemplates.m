function prompt = promptTemplates(record, cfg, feeder, level, variant)
%PROMPTTEMPLATES  Build the domain-knowledge-enhanced prompt. Core innovation #3.
%
%   PROMPT = PROMPTTEMPLATES(RECORD, CFG, FEEDER, LEVEL, VARIANT)
%
%   LEVEL   'summary'  one sentence, for shift dispatchers
%           'standard' two to three sentences, for system engineers
%           'detailed' a paragraph, for planning staff
%
%   VARIANT 'full'      grid context + state + action + SHAP + sensitivity
%           'noshap'    ablation 1: attribution withheld
%           'nodomain'  ablation 2: generic prompt, no topology or limits
%           'shaponly'  raw SHAP numbers with no narrative scaffolding
%
%   WHAT MAKES THIS "DOMAIN-KNOWLEDGE-ENHANCED"
%   -------------------------------------------
%   Four classes of grid knowledge are injected that a bare attribution
%   vector does not carry:
%     1. TOPOLOGY  -- which lateral each device sits on, and its electrical
%        distance from the substation, so the model can say "at the far end
%        of Lateral C" instead of "at bus 30".
%     2. LIMITS    -- the ANSI C84.1 band and how close the feeder is to it,
%        so the explanation can state urgency truthfully.
%     3. SENSITIVITY -- the measured dV/dQ at THIS operating point, so the
%        "expected effect" clause is a quantitative prediction that
%        COUNTERFACTUALTEST can later falsify, rather than a plausible
%        guess. This is the single most important element: it converts the
%        explanation from narrative into a checkable claim.
%     4. ALTERNATIVES -- the devices that were available but not moved,
%        which is what makes the explanation contrastive. Operators almost
%        never ask "why this?"; they ask "why this rather than that?".
%
%   ANTI-HALLUCINATION INSTRUCTIONS
%   -------------------------------
%   The task block explicitly forbids inventing numbers and instructs the
%   model to flag a diffuse attribution rather than manufacture a dominant
%   cause. An LLM asked to explain a decision will always produce a
%   confident explanation, including when the attribution says there is no
%   dominant driver; that failure mode is what an operator will eventually
%   catch, and it destroys trust in the whole tool.
%
%   See also GENERATEEXPLANATION, VOLTAGESENSITIVITY.

if nargin < 4 || isempty(level),   level   = 'standard'; end
if nargin < 5 || isempty(variant), variant = 'full';     end

Vm   = record.state.Vm(:);
info = record.info;

% =====================================================================
% Block 1: grid context
% =====================================================================
if strcmpi(variant, 'nodomain')
    gridBlock = sprintf(['You are an assistant. Explain why an automated ', ...
        'controller took the following action.\n\n']);
else
    capList = '';
    for k = 1:cfg.dims.nCap
        b = cfg.cap.buses(k);
        capList = [capList sprintf('    - Bus %2d (%s, %d hops from substation): %d kvar bank\n', ...
            b, feeder.zoneName{feeder.zone(b)}, feeder.depth(b), cfg.cap.kvar(k))]; %#ok<AGROW>
    end
    pvList = '';
    for k = 1:cfg.dims.nPV
        b = cfg.pv.buses(k);
        pvList = [pvList sprintf('    - Bus %2d (%s, %d hops): %d kW PV, inverter rated %.0f kVA\n', ...
            b, feeder.zoneName{feeder.zone(b)}, feeder.depth(b), ...
            cfg.pv.ratedKW(k), cfg.pv.ratedKW(k)/cfg.pv.pfMin)]; %#ok<AGROW>
    end

    gridBlock = sprintf([ ...
        'You are a distribution system dispatch expert writing for a control-room\n' ...
        'operator. Explain, in plain English, why the automatic Volt/VAR controller\n' ...
        'took the action described below.\n\n' ...
        '[GRID CONTEXT]\n' ...
        '%s' ...
        '  Substation: bus 1, on-load tap changer, %+d to %+d taps of %.2f%% each\n' ...
        '  Voltage standard: ANSI C84.1 Range A, %.2f-%.2f p.u.\n' ...
        '  Capacitor banks:\n%s' ...
        '  PV inverters:\n%s' ...
        '  Physical note: on a radial feeder, voltage falls with distance from the\n' ...
        '  substation under load, and can RISE above nominal when distributed PV\n' ...
        '  exports more power than the local load consumes (reverse power flow).\n\n'], ...
        networkBlock(cfg), ...
        cfg.oltc.tapMin, cfg.oltc.tapMax, cfg.oltc.stepPU*100, ...
        cfg.limits.Vmin, cfg.limits.Vmax, capList, pvList);
end

% =====================================================================
% Block 2: current state
% =====================================================================
[vMin, iMin] = min(Vm);
[vMax, iMax] = max(Vm);
nViol = nnz(Vm < cfg.limits.Vmin | Vm > cfg.limits.Vmax);

pvTotalMW = sum(record.state.pvP) * cfg.net.baseMVA;
pvRatedMW = sum(cfg.pv.ratedKW) / 1000;
loadMW    = sum(record.state.Pload) * cfg.net.baseMVA;

if nViol > 0
    violStr = sprintf('%d bus(es) OUTSIDE the statutory band', nViol);
else
    violStr = 'all buses within the statutory band';
end

stateBlock = sprintf([ ...
    '[CURRENT STATE - before the action]\n' ...
    '  Time: hour %.0f of the operating day (%s scenario)\n' ...
    '  Total feeder load: %.2f MW\n' ...
    '  Total PV output: %.2f MW (%.0f%% of %.1f MW rated)\n' ...
    '  Lowest voltage:  %.4f p.u. at bus %d\n' ...
    '  Highest voltage: %.4f p.u. at bus %d\n' ...
    '  Status: %s\n\n'], ...
    record.hour, record.scenario, loadMW, pvTotalMW, ...
    100*pvTotalMW/max(pvRatedMW,eps), pvRatedMW, ...
    vMin, iMin, vMax, iMax, violStr);

% =====================================================================
% Block 3: the decision
% =====================================================================
actionBlock = sprintf([ ...
    '[CONTROLLER DECISION]\n' ...
    '  %s\n' ...
    '  Devices moved this interval: %d\n\n'], ...
    record.actionText, record.info.interpBreakdown.nMoved);

% =====================================================================
% Block 4: SHAP attribution
% =====================================================================
if strcmpi(variant, 'noshap')
    shapBlock = '';
else
    shapBlock = sprintf('[FEATURE ATTRIBUTION - Shapley values for the %s channel]\n', ...
        record.shapMeta.explainedChannelName);
    for i = 1:numel(record.topFeatures)
        tf = record.topFeatures(i);
        shapBlock = [shapBlock sprintf('  %d. %-40s %+.4f  (%.0f%% of total attribution)\n', ...
            i, tf.name, tf.value, 100*tf.share)]; %#ok<AGROW>
    end
    topShare = record.topFeatures(1).share;
    if topShare < 0.25
        shapBlock = [shapBlock sprintf( ...
            ['  NOTE: no single factor dominates (top factor only %.0f%%). This\n' ...
             '  decision is driven by a combination of conditions.\n'], 100*topShare)];
    end
    shapBlock = [shapBlock sprintf('\n')];
end

% =====================================================================
% Block 5: sensitivity and alternatives (the contrastive material)
% =====================================================================
if strcmpi(variant, 'nodomain') || strcmpi(variant, 'shaponly')
    physBlock = '';
else
    physBlock = buildPhysicsBlock(record, cfg, feeder);
end

% =====================================================================
% Block 6: task
% =====================================================================
switch lower(level)
    case 'summary'
        taskBlock = [ ...
            '[TASK]\n' ...
            'Write ONE sentence, maximum 25 words, that a shift dispatcher can read\n' ...
            'at a glance. State what was done and the single main reason. No preamble.\n'];
    case 'standard'
        taskBlock = [ ...
            '[TASK]\n' ...
            'Write 2-3 sentences for a system engineer covering:\n' ...
            '  (1) why this action was taken;\n' ...
            '  (2) which factors were decisive, referring to the attribution above;\n' ...
            '  (3) the expected effect on the feeder.\n'];
    case 'detailed'
        taskBlock = [ ...
            '[TASK]\n' ...
            'Write one paragraph (120-180 words) for a planning engineer covering:\n' ...
            '  (1) the causal chain from system conditions to this action;\n' ...
            '  (2) the decisive factors and their relative weight;\n' ...
            '  (3) the quantitative expected effect;\n' ...
            '  (4) which alternative actions were available and why this one fits better;\n' ...
            '  (5) any residual risk if conditions change.\n'];
    otherwise
        error('promptTemplates:badLevel', ...
            'LEVEL must be summary, standard or detailed.');
end

taskBlock = [taskBlock ...
    '\n[RULES]\n' ...
    '  - Use ONLY the numbers given above. Do not invent voltages, ratings or buses.\n' ...
    '  - If the attribution shows no dominant factor, say so rather than\n' ...
    '    presenting one factor as the cause.\n' ...
    '  - Write for a professional operator: direct, no hedging, no bullet points.\n' ...
    '  - Do not repeat the input blocks back; output only the explanation text.\n'];

prompt = [gridBlock stateBlock actionBlock shapBlock physBlock taskBlock];
end

% =====================================================================
function blk = buildPhysicsBlock(record, cfg, feeder)
%BUILDPHYSICSBLOCK  Predicted effect and unused alternatives.

S = record.sensitivity;
bd = record.info.interpBreakdown;

blk = sprintf('[MEASURED SENSITIVITY AT THIS OPERATING POINT]\n');

% Predicted voltage effect of each device that actually moved.
nCap = cfg.dims.nCap;
movedIdx = find(bd.moved);
if isempty(movedIdx)
    blk = [blk sprintf('  No device moved, so no voltage change is expected.\n')];
else
    for k = movedIdx(:)'
        if k == 1
            dV = S.dVdT * (record.dev.tap - tapOf(record.devPrev));
            blk = [blk sprintf(['  OLTC tap change: predicted +%.4f p.u. at the substation,\n' ...
                '    %+.4f p.u. at the feeder end (bus %d).\n'], ...
                dV(1), dV(feederEnd(cfg)), feederEnd(cfg))]; %#ok<AGROW>
        else
            j = k - 1;                      % index into the reactive device list
            dQ = bd.dQ(k);
            dV = S.dVdQ(:, j) * dQ;
            [mx, bmx] = max(abs(dV));
            blk = [blk sprintf(['  %s: predicted %+.4f p.u. locally, largest change\n' ...
                '    %+.4f p.u. at bus %d.\n'], ...
                S.devName{j}, dV(S.devBus(j)), sign(dV(bmx))*mx, bmx)]; %#ok<AGROW>
        end
    end
end

% Contrastive material: the strongest device that was NOT used.
notMoved = find(~bd.moved);
notMoved = notMoved(notMoved > 1);
if ~isempty(notMoved)
    eff = zeros(size(notMoved));
    for i = 1:numel(notMoved)
        j = notMoved(i) - 1;
        eff(i) = max(abs(S.dVdQ(:, j)));
    end
    [~, best] = max(eff);
    j = notMoved(best) - 1;
    blk = [blk sprintf(['  Alternative NOT used: %s, which has a comparable\n' ...
        '    sensitivity of %.4f p.u. per p.u. VAr.\n'], S.devName{j}, eff(best))];
end

% Why the controller held back, if the interpretability penalty bit.
if bd.total > 0.5
    blk = [blk sprintf(['  The controller is penalised for moving many or electrically\n' ...
        '    distant devices at once; this interval scored %.2f on that measure.\n'], ...
        bd.total)];
end

blk = [blk sprintf('\n')];
end

function t = tapOf(devPrev)
if isempty(devPrev) || ~isfield(devPrev,'tap')
    t = 0;
else
    t = devPrev.tap;
end
end

% =====================================================================
function s = networkBlock(cfg)
%NETWORKBLOCK  Network description for the [GRID CONTEXT] block.
%   The 33-bus text is the default and is unchanged; other feeders set
%   cfg.net.description (see XRLCONFIG69).
if isfield(cfg.net, 'description') && ~isempty(cfg.net.description)
    s = cfg.net.description;
else
    s = sprintf([ ...
        '  Network: IEEE 33-bus radial distribution feeder, %.2f kV, %.0f MVA base\n' ...
        '  Structure: one main feeder (buses 1-18) and three laterals\n' ...
        '             (19-22 off bus 2, 23-25 off bus 3, 26-33 off bus 6)\n'], ...
        cfg.net.baseKV, cfg.net.baseMVA);
end
end

function b = feederEnd(cfg)
b = 18;
if isfield(cfg.net, 'feederEndBus'), b = cfg.net.feederEndBus; end
end
