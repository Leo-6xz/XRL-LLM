function agent = createPPOAgent(obsInfo, actInfo, cfg)
%CREATEPPOAGENT  Build a PPO agent for continuous VVC setpoints.
%
%   AGENT = CREATEPPOAGENT(OBSINFO, ACTINFO, CFG)
%
%   HYPERPARAMETERS
%   ---------------
%   All read from CFG.PPO; see XRLCONFIG for values and rationale. The two
%   that matter most here:
%     entropyWeight (0.01) -- with 9 continuous channels and a quantised
%         environment, the reward surface has plateaus at every
%         quantisation boundary. Without an entropy bonus the policy
%         collapses onto a deterministic corner early and stops exploring
%         capacitor combinations.
%     actorLR (1e-4) an order of magnitude below criticLR (1e-3) -- the
%         usual PPO ratio; a critic that lags the actor produces badly
%         scaled advantages and unstable clipping.
%
%   See also XRL33ENV, TRAINPPO.

if nargin < 3 || isempty(cfg)
    cfg = xrlConfig();
end

if ~license('test','RL_Toolbox') && isempty(ver('rl'))
    error('createPPOAgent:noToolbox', ...
        ['Reinforcement Learning Toolbox is required to build a PPO agent. ', ...
         'The rest of the pipeline (SHAP, LLM, evaluation, dashboard) runs ', ...
         'without it -- use RULEPOLICY as the policy instead.']);
end

nObs = obsInfo.Dimension(1);
nAct = actInfo.Dimension(1);
h    = cfg.ppo.hidden;

% =====================================================================
% CRITIC: V(s)
% =====================================================================
% Observation normalization. The raw vector mixes voltages (~1.0) with
% per-bus loads (~0.005-0.04 p.u.), a three-order-of-magnitude spread that
% leaves the first layer badly conditioned and the load channels
% effectively invisible. NORM.SCALE and NORM.BIAS map every channel to
% roughly zero mean and unit range using known physical bounds, so no
% training-data statistics are needed.
norm = observationNormalization(cfg);

criticNet = [
    featureInputLayer(nObs, 'Name', 'obs')
    scalingLayer('Name','cnorm','Scale',norm.scale,'Bias',norm.bias)
    fullyConnectedLayer(h(1), 'Name', 'cfc1')
    reluLayer('Name', 'crelu1')
    fullyConnectedLayer(h(2), 'Name', 'cfc2')
    reluLayer('Name', 'crelu2')
    fullyConnectedLayer(1, 'Name', 'value')];

criticNet = dlnetwork(criticNet);
critic = rlValueFunction(criticNet, obsInfo, ...
    'ObservationInputNames', 'obs');

% =====================================================================
% ACTOR: Gaussian policy with separate mean and standard-deviation heads
% =====================================================================
lower = actInfo.LowerLimit;
upper = actInfo.UpperLimit;
if isscalar(lower), lower = lower * ones(nAct,1); end
if isscalar(upper), upper = upper * ones(nAct,1); end
scaleVec = (upper - lower) / 2;
biasVec  = (upper + lower) / 2;

trunk = [
    featureInputLayer(nObs, 'Name', 'obs')
    scalingLayer('Name','anorm','Scale',norm.scale,'Bias',norm.bias)
    fullyConnectedLayer(h(1), 'Name', 'afc1')
    reluLayer('Name', 'arelu1')
    fullyConnectedLayer(h(2), 'Name', 'afc2')
    reluLayer('Name', 'arelu2')];

% Small initial weights on the mean head keep the policy near the NEUTRAL
% action at episode 1 and well away from the tanh saturation region. A
% default-initialised head starts with pre-activations large enough that
% tanh is already flat on some channels, and a single bad update can pin
% it there permanently.
meanPath = [
    fullyConnectedLayer(nAct, 'Name', 'meanFC', ...
        'WeightsInitializer', @(sz) 0.01*randn(sz), ...
        'BiasInitializer',    @(sz) zeros(sz))
    tanhLayer('Name', 'meanTanh')
    scalingLayer('Name', 'meanScale', 'Scale', scaleVec, 'Bias', biasVec)];

% Softplus keeps sigma strictly positive. The small additive floor stops
% the policy from collapsing to a delta function, which would make the
% log-probability ratio blow up and destabilise the clipped objective.
% Bias the std head so the initial standard deviation is ~0.4 rather than
% softplus(0) = 0.69. At 0.69 on a [-1,1] action space most samples clip
% at the bounds, which wastes exploration and biases the gradient.
stdPath = [
    fullyConnectedLayer(nAct, 'Name', 'stdFC', ...
        'WeightsInitializer', @(sz) 0.01*randn(sz), ...
        'BiasInitializer',    @(sz) -0.71*ones(sz))
    softplusLayer('Name', 'stdSoftplus')];

actorNet = layerGraph(trunk);
actorNet = addLayers(actorNet, meanPath);
actorNet = addLayers(actorNet, stdPath);
actorNet = connectLayers(actorNet, 'arelu2', 'meanFC');
actorNet = connectLayers(actorNet, 'arelu2', 'stdFC');

actorNet = dlnetwork(actorNet);

actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs', ...
    'ActionMeanOutputNames', 'meanScale', ...
    'ActionStandardDeviationOutputNames', 'stdSoftplus');

% =====================================================================
% Optimizer and PPO options
% =====================================================================
% OPTIMISER OPTIONS BELONG ON THE AGENT OPTIONS, NOT THE APPROXIMATOR.
%
% An earlier version of this file tried
%     actor.OptimizerOptions = rlOptimizerOptions(...)
% which is not a property of rlContinuousGaussianActor. The assignment
% threw, a try/catch swallowed it as a warning, and BOTH networks silently
% trained at MATLAB's default learn rate of 0.01 -- 100x the intended
% actor rate. The result was total policy collapse: the tanh mean path
% saturated at +1 within a few hundred episodes, gradients vanished, and
% the agent froze holding every device at maximum injection. Measured
% consequence: switching per step 0.000, 97.5% of bus-steps in
% overvoltage, and 283 kW of loss against 80 kW uncontrolled.
%
% The lesson is not just the API. A misconfigured learn rate is silent --
% training completes, the reward curve looks merely "flat" rather than
% broken, and only the switching statistic reveals it. Never let an
% optimiser assignment fail as a warning.
agentOpts = rlPPOAgentOptions( ...
    'ExperienceHorizon',       cfg.ppo.experienceHorizon, ...
    'MiniBatchSize',           cfg.ppo.miniBatchSize, ...
    'ClipFactor',              cfg.ppo.clipFactor, ...
    'EntropyLossWeight',       cfg.ppo.entropyWeight, ...
    'NumEpoch',                cfg.ppo.numEpoch, ...
    'AdvantageEstimateMethod', 'gae', ...
    'GAEFactor',               cfg.ppo.gaeFactor, ...
    'SampleTime',              cfg.ppo.sampleTime, ...
    'DiscountFactor',          cfg.ppo.discountFactor);

agentOpts.ActorOptimizerOptions = rlOptimizerOptions( ...
    'LearnRate',         cfg.ppo.actorLR, ...
    'GradientThreshold', cfg.ppo.maxGradientNorm);
agentOpts.CriticOptimizerOptions = rlOptimizerOptions( ...
    'LearnRate',         cfg.ppo.criticLR, ...
    'GradientThreshold', cfg.ppo.maxGradientNorm);

% Verify rather than assume. If a future release moves these properties
% again, this must stop the run, not warn and continue.
assert(abs(agentOpts.ActorOptimizerOptions.LearnRate - cfg.ppo.actorLR) < 1e-12, ...
    'createPPOAgent:actorLR', 'Actor learn rate did not take effect.');
assert(abs(agentOpts.CriticOptimizerOptions.LearnRate - cfg.ppo.criticLR) < 1e-12, ...
    'createPPOAgent:criticLR', 'Critic learn rate did not take effect.');

agent = rlPPOAgent(actor, critic, agentOpts);

if cfg.io.verbose
    fprintf('  PPO agent: actor LR %.1e, critic LR %.1e, entropy %.3f, clip %.2f\n', ...
        agentOpts.ActorOptimizerOptions.LearnRate, ...
        agentOpts.CriticOptimizerOptions.LearnRate, ...
        agentOpts.EntropyLossWeight, agentOpts.ClipFactor);
end
end
