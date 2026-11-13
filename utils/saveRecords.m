function outFile = saveRecords(records, explanations, cfg, tag)
%SAVERECORDS  Persist attribution records as .mat and a readable JSON log.
%
%   OUTFILE = SAVERECORDS(RECORDS, EXPLANATIONS, CFG, TAG)
%
%   Writes two artefacts to cfg.io.resultsDir:
%     <tag>_records.mat   full records, including state structs
%     <tag>_records.json  the attribution-action mapping table of Step 3.3
%
%   The JSON deliberately omits the state structs and keeps only what the
%   brief's schema specifies plus the generated explanation. It is the file
%   to attach to a paper's supplementary material: readable without MATLAB,
%   and small enough to inspect by hand.

if nargin < 4 || isempty(tag), tag = 'xrl'; end
if ~exist(cfg.io.resultsDir,'dir'), mkdir(cfg.io.resultsDir); end

matFile = fullfile(cfg.io.resultsDir, sprintf('%s_records.mat', tag));
save(matFile, 'records', 'explanations', 'cfg', '-v7.3');

n = numel(records);
entries = cell(n,1);
for i = 1:n
    r = records(i);
    e = struct();
    e.timestamp = datestr(r.timestamp, 'yyyy-mm-ddTHH:MM:SS');
    e.episode   = r.episode;
    e.step      = r.t;
    e.hour      = r.hour;
    e.scenario  = r.scenario;
    e.action    = r.action(:)';
    e.actionText= r.actionText;
    e.explainedChannel = r.shapMeta.explainedChannelName;

    sv = struct();
    for j = 1:numel(r.playerNames)
        sv.(matlab.lang.makeValidName(r.playerNames{j})) = r.phi(j);
    end
    e.shap_values = sv;

    e.top_features = arrayfun(@(t) struct( ...
        'name', t.name, 'value', t.value, 'share', t.share), ...
        r.topFeatures, 'UniformOutput', false);

    e.reward     = r.reward;
    e.Vmin       = r.info.Vmin;
    e.Vmax       = r.info.Vmax;
    e.lossKW     = r.info.PlossKW;
    e.nViolated  = r.info.nViolatedBuses;

    if ~isempty(explanations) && i <= numel(explanations)
        e.explanation = explanations{i};
    end
    entries{i} = e;
end

jsonFile = fullfile(cfg.io.resultsDir, sprintf('%s_records.json', tag));
fid = fopen(jsonFile, 'w');
if fid < 0
    warning('saveRecords:noWrite','Could not open %s for writing.', jsonFile);
else
    fprintf(fid, '%s', jsonencode(entries, 'PrettyPrint', true));
    fclose(fid);
end

outFile = matFile;
if cfg.io.verbose
    fprintf('  Saved %d records to %s and %s\n', n, matFile, jsonFile);
end
end
