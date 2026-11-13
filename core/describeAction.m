function [txt, parts, brief] = describeAction(dev, devPrev, cfg)
%DESCRIBEACTION  Turn a decoded device struct into an operator sentence.
%
%   [TXT, PARTS] = DESCRIBEACTION(DEV, DEVPREV, CFG) produces text such as
%
%     "Raised OLTC tap from 0 to +2 (slack 1.000 -> 1.025 p.u.); closed
%      capacitor bank at bus 30 (+300 kvar); set PV inverter at bus 17 to
%      +0.18 Mvar (absorbing 0.00 Mvar previously)."
%
%   Only devices that actually CHANGED are described. This matters: an
%   explanation that lists all nine channels every step trains operators
%   to ignore it. PARTS returns the individual clauses so the dashboard
%   can render them as separate rows.
%
%   BRIEF is a condensed one-clause form for the dispatcher-level summary,
%   e.g. "raised the tap one step and closed 3 capacitor banks". The full
%   TXT is unsuitable there: when several devices move at once it runs to
%   five clauses and 60 words, which defeats the purpose of a one-line
%   summary an operator reads at a glance.
%
%   See also DECODEACTION, GENERATEEXPLANATION.

if isempty(devPrev)
    devPrev.tap  = 0;
    devPrev.capQ = zeros(size(dev.capQ));
    devPrev.invQ = zeros(size(dev.invQ));
    devPrev.capOn= zeros(size(dev.capOn));
    devPrev.V0   = cfg.oltc.nominalPU;
end

baseMVA = cfg.net.baseMVA;
parts = {};

% ---- OLTC ------------------------------------------------------------
if dev.tap ~= devPrev.tap
    dirWord = 'Raised';
    if dev.tap < devPrev.tap
        dirWord = 'Lowered';
    end
    parts{end+1} = sprintf('%s OLTC tap from %+d to %+d (substation voltage %.4f -> %.4f p.u.)', ...
        dirWord, devPrev.tap, dev.tap, devPrev.V0, dev.V0);
end

% ---- Capacitor banks -------------------------------------------------
for k = 1:numel(dev.capQ)
    if abs(dev.capQ(k) - devPrev.capQ(k)) < 1e-9
        continue
    end
    b     = dev.capBuses(k);
    dkvar = (dev.capQ(k) - devPrev.capQ(k)) * baseMVA * 1000;
    if dev.capQ(k) > devPrev.capQ(k)
        parts{end+1} = sprintf('closed capacitor bank at bus %d (%+.0f kvar)', b, dkvar); %#ok<AGROW>
    else
        parts{end+1} = sprintf('opened capacitor bank at bus %d (%+.0f kvar)', b, dkvar); %#ok<AGROW>
    end
end

% ---- PV inverters ----------------------------------------------------
for k = 1:numel(dev.invQ)
    if abs(dev.invQ(k) - devPrev.invQ(k)) < 0.005
        continue
    end
    b    = dev.pvBuses(k);
    qNew = dev.invQ(k) * baseMVA;
    qOld = devPrev.invQ(k) * baseMVA;
    if qNew >= 0
        mode = 'injecting';
    else
        mode = 'absorbing';
    end
    parts{end+1} = sprintf('set PV inverter at bus %d to %+.3f Mvar (%s; was %+.3f Mvar)', ...
        b, qNew, mode, qOld); %#ok<AGROW>
end

if isempty(parts)
    txt   = 'No device change: all setpoints held at their previous positions.';
    brief = 'held all setpoints';
    return
end

% ---- Condensed form -------------------------------------------------
bits = {};
dTap = dev.tap - devPrev.tap;
if dTap > 0
    bits{end+1} = sprintf('raised the tap %d step(s)', dTap);
elseif dTap < 0
    bits{end+1} = sprintf('lowered the tap %d step(s)', -dTap);
end
nClosed = nnz(dev.capQ - devPrev.capQ >  1e-9);
nOpened = nnz(dev.capQ - devPrev.capQ < -1e-9);
if nClosed > 0, bits{end+1} = sprintf('closed %d capacitor bank(s)', nClosed); end
if nOpened > 0, bits{end+1} = sprintf('opened %d capacitor bank(s)', nOpened); end
nInv = nnz(abs(dev.invQ - devPrev.invQ) >= 0.005);
if nInv > 0, bits{end+1} = sprintf('adjusted %d PV inverter(s)', nInv); end

if isempty(bits)
    brief = 'made a minor setpoint adjustment';
elseif numel(bits) == 1
    brief = bits{1};
else
    brief = [strjoin(bits(1:end-1), ', ') ' and ' bits{end}];
end

% Capitalise the first clause and join.
first = parts{1};
first(1) = upper(first(1));
if numel(parts) == 1
    txt = [first '.'];
else
    txt = [first '; ' strjoin(parts(2:end), '; ') '.'];
end
end
