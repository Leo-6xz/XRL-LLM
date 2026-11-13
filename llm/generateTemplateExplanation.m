function txt = generateTemplateExplanation(record, cfg, feeder, level, variant)
%GENERATETEMPLATEEXPLANATION  Strong non-LLM baseline explanation generator.
%
%   TXT = GENERATETEMPLATEEXPLANATION(RECORD, CFG, FEEDER, LEVEL)
%
%   This is ablation 5's control condition ("template-based generation vs.
%   LLM generation"). It is written to be the STRONGEST reasonable
%   template, not a straw man: it consumes the same SHAP attributions, the
%   same topology and the same measured sensitivities as the LLM prompt.
%   The only thing it lacks is free-form language.
%
%   Building the baseline this way is what makes the eventual comparison
%   worth reporting. A template that merely prints "action taken because
%   feature X = 0.42" would lose to any LLM and the resulting effect size
%   would say nothing about language models.
%
%   It also serves as the offline fallback, so the pipeline produces real
%   explanations with no model installed.

if nargin < 4 || isempty(level),   level   = 'standard'; end
if nargin < 5 || isempty(variant), variant = 'full';     end

% Honour the ablation variant. Without this, the "no SHAP" and "no domain
% knowledge" conditions produce text IDENTICAL to the full condition
% whenever the template backend is in use, and the ablation silently
% reports a zero effect for every prompt-composition condition.
useSHAP   = ~strcmpi(variant, 'noshap');
useDomain = ~strcmpi(variant, 'nodomain');

Vm = record.state.Vm(:);
[vMin, iMin] = min(Vm);
[vMax, iMax] = max(Vm);
nViol = nnz(Vm < cfg.limits.Vmin | Vm > cfg.limits.Vmax);

tf   = record.topFeatures(1);
zone = feeder.zoneName{feeder.zone(iMin)};

% ---- Trigger clause: what condition prompted the action --------------
if nViol > 0 && vMin < cfg.limits.Vmin
    trigger = sprintf('%d bus(es) had fallen below the %.2f p.u. limit, the worst being bus %d at %.4f p.u. on the %s', ...
        nViol, cfg.limits.Vmin, iMin, vMin, lower(zone));
elseif nViol > 0 && vMax > cfg.limits.Vmax
    trigger = sprintf('%d bus(es) had risen above the %.2f p.u. limit, the worst being bus %d at %.4f p.u., consistent with PV export exceeding local load', ...
        nViol, cfg.limits.Vmax, iMax, vMax);
elseif vMin < cfg.limits.VsoftMin
    trigger = sprintf('the feeder was approaching the lower limit (bus %d at %.4f p.u.) with no violation yet', iMin, vMin);
elseif vMax > cfg.limits.VsoftMax
    trigger = sprintf('the feeder was approaching the upper limit (bus %d at %.4f p.u.) with no violation yet', iMax, vMax);
else
    trigger = sprintf('all buses were inside the statutory band (%.4f to %.4f p.u.) and the adjustment was made to reduce losses', vMin, vMax);
end

% ---- Driver clause: what the attribution says ------------------------
if tf.share < 0.25
    driver = sprintf('No single factor dominated the decision: the largest contributor, %s, accounted for only %.0f%% of the total attribution, so this action reflects a combination of conditions', ...
        lower(tf.name), 100*tf.share);
else
    driver = sprintf('The decisive factor was %s, which carried %.0f%% of the total attribution', ...
        lower(tf.name), 100*tf.share);
end
if numel(record.topFeatures) >= 2
    driver = sprintf('%s, followed by %s at %.0f%%', driver, ...
        lower(record.topFeatures(2).name), 100*record.topFeatures(2).share);
end

% ---- Effect clause: the quantitative prediction ----------------------
effect = predictedEffect(record, cfg);

switch lower(level)
    case 'summary'
        % One sentence for a shift dispatcher. Uses the CONDENSED action
        % description: the full one runs to five clauses when several
        % devices move together, which is not a summary.
        if isfield(record,'actionBrief') && ~isempty(record.actionBrief)
            act = record.actionBrief;
        else
            [~, ~, act] = describeAction(record.dev, record.devPrev, cfg);
        end
        txt = sprintf('%s because %s.', capitalise(act), trigger);
        return

    case 'standard'
        txt = sprintf('%s The action was taken because %s.', ...
            capitalise(record.actionText), trigger);
        if useSHAP
            txt = sprintf('%s %s.', txt, driver);
        end
        if useDomain
            txt = sprintf('%s %s', txt, effect);
        end
        return

    case 'detailed'
        txt = sprintf('%s The action was taken because %s.', ...
            capitalise(record.actionText), trigger);
        if useSHAP
            txt = sprintf('%s %s.', txt, driver);
        end
        if useDomain
            txt = sprintf('%s %s %s %s', txt, effect, ...
                alternativeClause(record, cfg), riskClause(record, cfg, vMin, vMax));
        end
        return

    otherwise
        error('generateTemplateExplanation:badLevel', ...
            'LEVEL must be summary, standard or detailed.');
end
end

% =====================================================================
function s = predictedEffect(record, cfg)
S  = record.sensitivity;
bd = record.info.interpBreakdown;
movedIdx = find(bd.moved);
if isempty(movedIdx)
    s = 'No device was moved, so no voltage change is expected this interval.';
    return
end
k = movedIdx(1);
if k == 1
    dV = S.dVdT * (record.dev.tap - tapOf(record.devPrev));
    fe = 18;
    if isfield(cfg.net, 'feederEndBus'), fe = cfg.net.feederEndBus; end
    s = sprintf('The tap change is expected to shift voltage by %+.4f p.u. at the substation and %+.4f p.u. at the feeder end (bus %d).', dV(1), dV(fe), fe);
else
    j  = k - 1;
    dV = S.dVdQ(:, j) * bd.dQ(k);
    [mx, b] = max(abs(dV));
    s = sprintf('Based on the measured sensitivity at this operating point, the change is expected to move voltage by %+.4f p.u. at bus %d and by %+.4f p.u. at bus %d, the most affected point.', ...
        dV(S.devBus(j)), S.devBus(j), sign(dV(b))*mx, b);
end
end

% =====================================================================
function s = alternativeClause(record, cfg)
S  = record.sensitivity;
bd = record.info.interpBreakdown;
notMoved = find(~bd.moved); notMoved = notMoved(notMoved > 1);
if isempty(notMoved)
    s = 'Every available device was already committed.';
    return
end
eff = arrayfun(@(k) max(abs(S.dVdQ(:, k-1))), notMoved);
[bestEff, i] = max(eff);
s = sprintf('The main alternative, %s, was left unchanged despite a comparable sensitivity of %.4f p.u. per p.u. VAr.', ...
    S.devName{notMoved(i)-1}, bestEff);
end

% =====================================================================
function s = riskClause(record, cfg, vMin, vMax)
head = cfg.limits.Vmax - vMax;
foot = vMin - cfg.limits.Vmin;
if foot < head
    s = sprintf('Residual risk: only %.4f p.u. of margin remains to the lower limit, so a further load increase would require additional support.', foot);
else
    s = sprintf('Residual risk: only %.4f p.u. of margin remains to the upper limit, so rising PV output would require reactive absorption.', head);
end
end

function t = tapOf(devPrev)
if isempty(devPrev) || ~isfield(devPrev,'tap'), t = 0; else, t = devPrev.tap; end
end

function s = capitalise(s)
if ~isempty(s), s(1) = upper(s(1)); end
end
