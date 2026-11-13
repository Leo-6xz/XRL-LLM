function [agent, info] = loadBestCheckpoint(cfg, trainingStats)
%LOADBESTCHECKPOINT  Recover the best policy reached during training.
%
%   [AGENT, INFO] = LOADBESTCHECKPOINT(CFG, TRAININGSTATS)
%
%   The agent saved at the end of TRAINPPO is the LAST one, which on a
%   stochastic-scenario environment is routinely not the best one. Measured
%   on two runs here:
%
%       run 1: best 50-episode average -2.553 at episode 1155,
%              final 50-episode average -4.643   (82% worse)
%
%   Discarding that is throwing away most of what training achieved. PPO
%   oscillates once it reaches the neighbourhood of a good policy, and with
%   no early stopping the final weights are essentially a random draw from
%   that oscillation.
%
%   This scans CFG.TRAIN.SAVEPATH's checkpoint directory, matches the file
%   whose episode number is closest to the best moving-average window, and
%   returns that agent.
%
%   If TRAININGSTATS is omitted it is loaded from CFG.TRAIN.SAVEPATH.
%
%   ALWAYS RE-EVALUATE. The best TRAINING average was measured on training
%   scenarios; confirm on held-out ones before reporting:
%
%       best = loadBestCheckpoint(cfg);
%       controlMetrics(makePolicyFcn(best,cfg), cfg, feeder, 60, ...
%                      struct('label','PPO best checkpoint'));
%
%   See also TRAINPPO, CONTROLMETRICS, DIAGNOSEPOLICY.

if nargin < 1 || isempty(cfg), cfg = xrlConfig(); end

if nargin < 2 || isempty(trainingStats)
    if ~exist(cfg.train.savePath,'file')
        error('loadBestCheckpoint:noRun', ...
            'No training run found at %s.', cfg.train.savePath);
    end
    S = load(cfg.train.savePath);
    trainingStats = S.trainingStats;
end

r = trainingStats.EpisodeReward(:);
w = min(cfg.train.scoreAveragingLen, numel(r));
m = movmean(r, [w-1 0]);
m(1:w-1) = -Inf;
[bestAvg, bestEp] = max(m);

ckptDir = fullfile(fileparts(cfg.train.savePath), 'checkpoints');
if ~exist(ckptDir,'dir')
    error('loadBestCheckpoint:noCheckpoints', ...
        ['No checkpoint directory at %s. Training must run with a ', ...
         'cfg.train.checkpointThreshold above the reward the agent reaches.'], ckptDir);
end

files = dir(fullfile(ckptDir, 'Agent*.mat'));
if isempty(files)
    error('loadBestCheckpoint:noCheckpoints', ...
        ['No Agent*.mat files in %s. Lower cfg.train.checkpointThreshold ', ...
         '(currently %g) so checkpoints are actually written.'], ...
        ckptDir, cfg.train.checkpointThreshold);
end

eps = zeros(numel(files),1);
for i = 1:numel(files)
    tok = regexp(files(i).name, 'Agent(\d+)\.mat', 'tokens', 'once');
    if isempty(tok), eps(i) = NaN; else, eps(i) = str2double(tok{1}); end
end
valid = ~isnan(eps);
if ~any(valid)
    error('loadBestCheckpoint:unparseable', ...
        'Could not parse episode numbers from the checkpoint filenames.');
end

[gap, k] = min(abs(eps(valid) - bestEp));
idx = find(valid); idx = idx(k);

C = load(fullfile(ckptDir, files(idx).name));
if isfield(C,'saved_agent'),   agent = C.saved_agent;
elseif isfield(C,'agent'),     agent = C.agent;
else
    fn = fieldnames(C);
    agent = C.(fn{1});
end

info.file          = fullfile(ckptDir, files(idx).name);
info.checkpointEp  = eps(idx);
info.bestEpisode   = bestEp;
info.bestAvgReward = bestAvg;
info.episodeGap    = gap;
info.finalAvgReward= mean(r(max(1,end-w+1):end));

if cfg.io.verbose
    fprintf(['  Best window: %.3f at episode %d | final: %.3f\n' ...
             '  Loaded checkpoint from episode %d (%s), gap %d episodes.\n'], ...
        bestAvg, bestEp, info.finalAvgReward, info.checkpointEp, ...
        files(idx).name, gap);
    if gap > 100
        warning('loadBestCheckpoint:farFromBest', ...
            ['Nearest checkpoint is %d episodes from the best window. ', ...
             'Lower cfg.train.checkpointThreshold for finer resolution.'], gap);
    end
end
end
