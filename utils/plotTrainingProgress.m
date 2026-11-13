function fig = plotTrainingProgress(trainingStats, cfg)
%PLOTTRAININGPROGRESS  Reward curve with a moving average.
%
%   FIG = PLOTTRAININGPROGRESS(TRAININGSTATS, CFG)
%
%   TRAININGSTATS is the output of TRAIN. The moving average is the curve
%   to read; raw episode reward on a stochastic-scenario environment is
%   dominated by which day was drawn, not by policy quality.
%
%   A flat average from the first episode usually means the policy
%   collapsed to a deterministic corner -- raise cfg.ppo.entropyWeight.
%   Violent oscillation means the actor learning rate is too high.

r = trainingStats.EpisodeReward(:);
n = numel(r);
w = min(cfg.train.scoreAveragingLen, max(1, floor(n/10)));

ma = movmean(r, [w-1 0]);

fig = figure('Position',[100 100 900 480],'Color','w');
plot(1:n, r, '-', 'Color',[0.75 0.80 0.88], 'LineWidth',0.8, ...
    'DisplayName','Episode reward');
hold on;
plot(1:n, ma, '-', 'Color',[0.10 0.35 0.75], 'LineWidth',2.0, ...
    'DisplayName',sprintf('Moving average (%d)', w));
hold off;
xlabel('Episode'); ylabel('Total episode reward');
title(sprintf('PPO training: %d episodes (alpha=%.3g beta=%.3g gamma=%.3g delta=%.3g)', ...
    n, cfg.reward.alpha, cfg.reward.beta, cfg.reward.gamma, cfg.reward.delta));
legend('Location','southeast'); grid on; box on;
set(gca,'FontSize',11);
end
