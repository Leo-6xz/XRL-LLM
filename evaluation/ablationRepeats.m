function results = ablationRepeats(records, llm, cfg, feeder, nRepeats, opts)
%ABLATIONREPEATS  Run the ablation R times and report ACROSS-RUN variability.
%
%   RESULTS = ABLATIONREPEATS(RECORDS, LLM, CFG, FEEDER, NREPEATS, OPTS)
%
%   WHY THIS EXISTS
%   ---------------
%   A single ablation run on this pipeline is not evidence. LLM generation
%   is stochastic even at low temperature, so repeating the identical
%   configuration produces different explanations and therefore different
%   scores. Measured on four runs of the quick configuration (n = 15):
%
%     condition    run A    run B    run C    run D     sign stable?  sig
%     noSHAP       +2.0%    -8.7%    -6.8%    +3.0%     NO -- FLIPS   1/4
%     noDomain    +15.1%    +6.7%    +2.7%    +5.7%     yes           1/4
%     shapOnly     -3.7%   -10.4%    -9.4%    -3.8%     yes           2/4
%     template     -8.4%   -15.6%   -13.1%    -9.8%     yes           4/4
%
%   The noSHAP effect changes SIGN across runs, and one run reported it as
%   significant at p = 0.039 -- a false positive that a single-run study
%   would have published. Only the template comparison survived all four.
%
%   The same data shows where the noise comes from. The llama3 judge scored
%   the identical configuration 4.56 then 4.27 (a 0.29 swing), while the
%   average gap BETWEEN judges was 0.10. Generation variance is roughly
%   twice the judge effect, so swapping judges without repeating runs
%   diagnoses the wrong problem -- which is exactly what happened here.
%
%   WHAT TO REPORT
%   --------------
%   RESULTS.SUMMARY gives, per condition, the mean gain across repeats, the
%   ACROSS-RUN standard deviation, the fraction of repeats sharing the
%   majority sign, and the fraction reaching p < 0.05. A condition that is
%   not sign-stable across repeats should not appear as a finding,
%   whatever its p-value in any single run.
%
%   COST
%   ----
%   NREPEATS x 5 conditions x numel(RECORDS) generations. With a local 8B
%   model at ~15 s each, 5 repeats over 15 records is on the order of an
%   hour. Run it once overnight rather than trusting a two-minute run.
%
%   Example
%   -------
%     cfg = xrlConfig();
%     cfg.llm.model     = 'llama3:8b';
%     cfg.llm.evalModel = 'mistral-nemo';
%     out = main('stages',{'shap'},'policy','rule','cfg',cfg);
%     llm = connectLLM(cfg);
%     rep = ablationRepeats(out.records, llm, cfg, out.feeder, 5);
%
%   See also ABLATIONEXPERIMENTS, GEVAL.

if nargin < 5 || isempty(nRepeats), nRepeats = 5; end
if nargin < 6 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nSamples'), opts.nSamples = min(50, numel(records)); end
if ~isfield(opts,'verbose'),  opts.verbose  = true; end

if ~llm.available
    warning('ablationRepeats:noLiveLLM', ...
        ['No live LLM. Repeats will be identical because the template ', ...
         'generator is deterministic, so this tells you nothing.']);
end

runs = cell(nRepeats,1);
for r = 1:nRepeats
    if opts.verbose
        fprintf('\n########## ablation repeat %d of %d ##########\n', r, nRepeats);
    end
    runs{r} = ablationExperiments(records, llm, cfg, feeder, ...
        struct('nSamples', opts.nSamples, 'verbose', opts.verbose));
end

condNames = {runs{1}.comparisons.vs};
nC = numel(condNames);

gain = nan(nRepeats, nC);
dval = nan(nRepeats, nC);
pval = nan(nRepeats, nC);
qFull= nan(nRepeats, 1);

for r = 1:nRepeats
    c = runs{r}.comparisons;
    for j = 1:nC
        k = find(strcmp({c.vs}, condNames{j}), 1);
        gain(r,j) = c(k).pctGain;
        dval(r,j) = c(k).cohensD;
        pval(r,j) = c(k).pValue;
    end
    fi = find(strcmp({runs{r}.conditions.name}, 'full'), 1);
    qFull(r) = runs{r}.conditions(fi).meanOverall;
end

results.nRepeats   = nRepeats;
results.condNames  = condNames;
results.gain       = gain;
results.cohensD    = dval;
results.pValue     = pval;
results.fullQuality= qFull;
results.runs       = runs;

results.summary = struct('condition',{},'meanGain',{},'sdGain',{}, ...
    'signStable',{},'majoritySignFrac',{},'sigFrac',{},'reportable',{});

for j = 1:nC
    g = gain(:,j);
    majority = sign(median(g));
    if majority == 0, majority = 1; end
    frac = mean(sign(g) == majority);

    s.condition        = condNames{j};
    s.meanGain         = mean(g);
    s.sdGain           = std(g);
    s.majoritySignFrac = frac;
    s.signStable       = (frac == 1);
    s.sigFrac          = mean(pval(:,j) < 0.05);
    % A finding is reportable only if every repeat agrees on the direction
    % AND a majority reach significance. Anything else is a single-run
    % artefact waiting to be contradicted by a referee's replication.
    s.reportable       = s.signStable && (s.sigFrac >= 0.5);
    results.summary(end+1) = s; %#ok<AGROW>
end

results.fullQualitySD = std(qFull);

if opts.verbose
    fprintf('\n================ ACROSS-RUN SUMMARY (%d repeats) ================\n', nRepeats);
    fprintf('  %-10s %10s %8s %12s %9s   %s\n', ...
        'condition','mean gain','SD','sign stable','p<.05','verdict');
    for j = 1:numel(results.summary)
        s = results.summary(j);
        if s.reportable, verdict = 'REPORTABLE'; else, verdict = 'not reportable'; end
        fprintf('  %-10s %9.1f%% %8.1f %11.0f%% %8.0f%%   %s\n', ...
            s.condition, s.meanGain, s.sdGain, 100*s.majoritySignFrac, ...
            100*s.sigFrac, verdict);
    end
    fprintf('\n  Full-condition quality across repeats: %.2f +/- %.2f\n', ...
        mean(qFull), results.fullQualitySD);
    fprintf(['  Compare that SD against the gaps above. Any effect smaller than\n' ...
             '  the run-to-run SD is not measurable at this sample size.\n']);
    fprintf('================================================================\n\n');
end
end
