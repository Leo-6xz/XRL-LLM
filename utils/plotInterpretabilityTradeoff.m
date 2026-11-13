function fig = plotInterpretabilityTradeoff(sweep, varargin)
%PLOTINTERPRETABILITYTRADEOFF  Fig. 6 -- interpretability weight trade-off.
%
%   FIG = PLOTINTERPRETABILITYTRADEOFF(SWEEP) plots mean active loss and
%   attribution concentration against the interpretability weight delta of
%   the reward (eq. 8), one trained agent per grid point, from the struct
%   returned by REWARDWEIGHTSWEEP -- i.e. STUDY.sweep from
%
%       study = runFullStudy('sweep', cfg);
%       fig   = plotInterpretabilityTradeoff(study.sweep);
%
%   or, after an overnight run, from the saved phase file:
%
%       S   = load(fullfile('results','phase_sweep_YYYYMMDD_HHMMSS.mat'));
%       fig = plotInterpretabilityTradeoff(S.sweep, ...
%                 'SavePath', fullfile('results','fig6_interpretability_tradeoff'));
%
%   OPTIONS
%     'SavePath'  path stem (no extension); writes <stem>.fig and <stem>.png
%                 at 300 dpi. Default '' (do not save).
%     'Title'     figure title. Default 'Interpretability weight trade-off'.
%
%   WHAT IT REPORTS
%     - Spearman rho between delta and mean active loss (self-contained; no
%       Statistics Toolbox, per the repository's design rule -- see
%       CHECKDEPENDENCIES and COUNTERFACTUALTEST). With a monotonic rise
%       across the grid this is +1.00.
%     - The knee delta* = argmax(concentration): the point beyond which more
%       interpretability weight buys loss but no further concentration.
%
%   The concentration series is sweep.topShare (mean top-1 attribution share,
%   eq. 24); the loss series is sweep.lossKW (mean active loss). Both are one
%   value per delta grid point. Field names are resolved defensively so this
%   also works if REWARDWEIGHTSWEEP names them slightly differently.
%
%   See also RUNFULLSTUDY, REWARDWEIGHTSWEEP, PLOTSHAPBAR.

ip = inputParser;
ip.addParameter('SavePath', '');
ip.addParameter('Title', 'Interpretability weight trade-off');
ip.parse(varargin{:});
opt = ip.Results;

% ---- resolve the three series -------------------------------------------
% deltaRange passed to rewardWeightSweep in runFullStudy('sweep'). Used as a
% fall-back only if the struct does not carry the grid back.
DEFAULT_DELTA = [0 0.02 0.05 0.10 0.20];

delta = pickField(sweep, {'delta','deltaRange','deltas','deltaGrid'});
if isempty(delta)
    delta = DEFAULT_DELTA;
    warning('plotFig6:assumedDelta', ...
        ['Delta grid not found on the sweep struct; assuming the grid used ', ...
         'by runFullStudy(''sweep''): [0 0.02 0.05 0.10 0.20]. Check that ', ...
         'this matches your run before quoting delta* in the caption.']);
end
loss = pickField(sweep, {'lossKW','meanLossKW','loss'});
conc = pickField(sweep, {'topShare','concentration','conc','meanTopShare'});

assert(~isempty(loss), 'plotFig6:noLoss', ...
    'Could not find a loss series (looked for lossKW/meanLossKW/loss).');
assert(~isempty(conc), 'plotFig6:noConc', ...
    'Could not find a concentration series (looked for topShare/concentration).');

% Flatten (alpha is held fixed in the sweep, so each is a 1-D series over
% delta even if returned as a 1 x nDelta grid) and sort by delta.
delta = delta(:).';  loss = loss(:).';  conc = conc(:).';
n = min([numel(delta) numel(loss) numel(conc)]);
delta = delta(1:n);  loss = loss(1:n);  conc = conc(1:n);
[delta, ord] = sort(delta);
loss = loss(ord);  conc = conc(ord);

% ---- statistics for the caption -----------------------------------------
rho = spearmanRho(delta, loss);          % loss vs delta, monotonicity
[concPeak, kPeak] = max(conc);
deltaStar = delta(kPeak);                 % the knee

% ---- plot ----------------------------------------------------------------
cLoss = [0.85 0.33 0.10];    % loss  (orange-red)
cConc = [0.00 0.45 0.74];    % concentration (blue)

fig = figure('Color','w','Position',[100 100 760 500]);

yyaxis left
hL = plot(delta, loss, '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'Color', cLoss, 'MarkerFaceColor', cLoss);
ylabel('Mean active loss (kW)');
ax = gca; ax.YColor = cLoss;

yyaxis right
hC = plot(delta, conc, '-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
    'Color', cConc, 'MarkerFaceColor', cConc);
ylabel('Attribution concentration  (mean top-1 share)');
ax.YColor = cConc;

hold on
% mark the knee on the concentration curve
hK = plot(deltaStar, concPeak, 'p', 'MarkerSize', 16, ...
    'MarkerFaceColor', [0.95 0.75 0.05], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
xline(deltaStar, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 1, ...
    'HandleVisibility', 'off');
text(deltaStar, concPeak, sprintf('  knee  \\delta^{*} = %.3g', deltaStar), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
    'FontWeight', 'bold');

xlabel('Interpretability weight  \delta   (eq. 8)');
title(opt.Title);
if exist('subtitle','file') || ~isempty(which('subtitle'))
    subtitle(sprintf(['loss vs \\delta: Spearman \\rho = %+.2f   |   ', ...
                      'concentration peaks at \\delta = %.3g'], rho, deltaStar));
end
grid on; box on
xlim([min(delta) - 0.01, max(delta) + 0.01]);

legend([hL hC hK], {'Mean active loss', 'Attribution concentration', 'Knee \delta^{*}'}, ...
    'Location', 'best');

% console echo so the numbers are copy-pasteable into the caption
fprintf('\n  Fig. 6 -- interpretability weight trade-off\n');
fprintf('    delta grid      : %s\n', mat2str(delta));
fprintf('    loss (kW)       : %s\n', num2str(loss, '%.1f  '));
fprintf('    concentration   : %s\n', num2str(conc, '%.3f  '));
fprintf('    Spearman rho(loss, delta) = %+.3f  (monotonic rise -> ~ +1)\n', rho);
fprintf('    knee delta*     = %.3g   (peak concentration %.3f)\n\n', deltaStar, concPeak);

% ---- save ----------------------------------------------------------------
if ~isempty(opt.SavePath)
    [d,~,~] = fileparts(opt.SavePath);
    if ~isempty(d) && ~exist(d,'dir'), mkdir(d); end
    try, savefig(fig, [opt.SavePath '.fig']); catch, end
    try
        exportgraphics(fig, [opt.SavePath '.png'], 'Resolution', 300);
    catch
        print(fig, [opt.SavePath '.png'], '-dpng', '-r300');
    end
    fprintf('  Saved: %s.fig / .png\n\n', opt.SavePath);
end
end

% =====================================================================
function val = pickField(s, names)
%PICKFIELD  First non-empty field of S among NAMES, else [].
val = [];
if ~isstruct(s), return; end
for i = 1:numel(names)
    if isfield(s, names{i}) && ~isempty(s.(names{i}))
        val = s.(names{i});
        return
    end
end
end

% =====================================================================
function rho = spearmanRho(x, y)
%SPEARMANRHO  Spearman rank correlation, self-contained (Pearson on tied
%   ranks). Implemented locally so the figure does not pull in the
%   Statistics Toolbox -- consistent with the rest of the repository.
rx = tiedRanks(x);
ry = tiedRanks(y);
rho = pearson(rx, ry);
end

function r = pearson(a, b)
a = a(:) - mean(a);  b = b(:) - mean(b);
den = sqrt((a.'*a) * (b.'*b));
if den < eps, r = NaN; else, r = (a.'*b) / den; end
end

function ranks = tiedRanks(v)
%TIEDRANKS  Ranks of V with ties assigned their average rank.
v = v(:).';
[~, ord] = sort(v);
ranks = zeros(size(v));
ranks(ord) = 1:numel(v);
u = unique(v);
for i = 1:numel(u)
    m = (v == u(i));
    if nnz(m) > 1, ranks(m) = mean(ranks(m)); end
end
end
