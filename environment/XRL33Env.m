classdef XRL33Env < rl.env.MATLABEnvironment
%XRL33ENV  Reinforcement Learning Toolbox wrapper for the IEEE 33-bus VVC task.
%
%   ENV = XRL33Env()             uses the default configuration
%   ENV = XRL33Env(CFG)          uses a configuration from XRLCONFIG
%   ENV = XRL33Env(CFG, FEEDER)  reuses a prebuilt feeder
%
%   This class is a THIN wrapper. All dynamics live in the toolbox-free
%   functions XRLRESET / XRLSTEP under core/. That separation is
%   deliberate: the SHAP layer, the LLM layer, the dashboard and every
%   unit test drive the environment through those functions, so they run
%   on a base MATLAB installation with no RL Toolbox licence. Only
%   training needs this file.
%
%   ACTION SPACE
%   ------------
%   9 continuous channels on [-1, 1]:
%     1     OLTC tap        (quantized to an integer tap inside DECODEACTION)
%     2:6   capacitor banks (thresholded to on/off inside DECODEACTION)
%     7:9   PV inverter Q   (scaled by the live reactive headroom)
%
%   The true problem is hybrid discrete/continuous. MATLAB has no native
%   hybrid PPO agent, so the policy emits a continuous vector and the
%   environment quantizes. See DECODEACTION for why this is sound and what
%   it costs.
%
%   OBSERVATION SPACE
%   -----------------
%   80 continuous channels; see BUILDOBSERVATION for the layout.
%
%   See also XRLRESET, XRLSTEP, CREATEPPOAGENT, TRAINPPO.

    properties
        Cfg      % configuration struct from XRLCONFIG
        Feeder   % feeder struct from BUILDFEEDER
    end

    properties (Access = protected)
        State        = []   % internal environment state (see XRLRESET)
        RngStream    = []   % episode scenario randomness
        LastInfo     = []   % reward decomposition of the most recent step
        EpisodeCount = 0
    end

    methods
        function this = XRL33Env(cfg, feeder)
            if nargin < 1 || isempty(cfg)
                cfg = xrlConfig();
            end
            if nargin < 2 || isempty(feeder)
                feeder = buildFeeder(cfg.net.dataFcn());
            end

            % --- Observation specification ---------------------------
            obsInfo = rlNumericSpec([cfg.dims.nObs 1]);
            obsInfo.Name = 'observations';
            obsInfo.Description = ...
                sprintf('%d bus voltages, %d active loads, %d PV outputs, load level, time of day, previous action', ...
                cfg.dims.nBus, cfg.dims.nBus-1, cfg.dims.nPV);

            % --- Action specification --------------------------------
            actInfo = rlNumericSpec([cfg.dims.nAction 1], ...
                'LowerLimit', -1, 'UpperLimit', 1);
            actInfo.Name = 'vvcSetpoints';
            actInfo.Description = 'OLTC tap, 5 capacitor banks, 3 inverter Q (all normalised)';

            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            this.Cfg       = cfg;
            this.Feeder    = feeder;
            this.RngStream = RandStream('mt19937ar', 'Seed', cfg.seed);
        end

        function [obs, reward, isDone, loggedSignals] = step(this, action)
            %STEP  Advance one control interval.
            if isempty(this.State)
                error('XRL33Env:notReset', 'Call reset(env) before step(env, a).');
            end

            [obs, reward, isDone, newState, info] = ...
                xrlStep(action, this.State, this.Cfg, this.Feeder);

            this.State    = newState;
            this.LastInfo = info;

            % LoggedSignals is how the RL Toolbox surfaces diagnostics to
            % the training monitor. Keeping the reward decomposed here is
            % what lets TRAINPPO plot the violation rate and the
            % interpretability penalty as separate curves -- without it,
            % a flat total-reward curve tells you nothing about which
            % objective stopped improving.
            loggedSignals.Vviol        = info.Vviol;
            loggedSignals.PlossKW      = info.PlossKW;
            loggedSignals.actionChange = info.actionChange;
            loggedSignals.interp       = info.interp;
            loggedSignals.nViolated    = info.nViolatedBuses;
            loggedSignals.Vmin         = info.Vmin;
            loggedSignals.Vmax         = info.Vmax;
            loggedSignals.scenario     = info.scenario;

            notifyEnvUpdated(this);
        end

        function obs = reset(this)
            %RESET  Draw a fresh 24-hour scenario.
            [obs, this.State] = xrlReset(this.Cfg, this.Feeder, this.RngStream);
            this.EpisodeCount = this.EpisodeCount + 1;
            this.LastInfo     = [];
            notifyEnvUpdated(this);
        end

        function s = getState(this)
            %GETSTATE  Expose the internal state (used by the dashboard).
            s = this.State;
        end

        function info = getLastInfo(this)
            %GETLASTINFO  Reward decomposition of the most recent step.
            info = this.LastInfo;
        end

        function setSeed(this, seed)
            %SETSEED  Reset scenario randomness for reproducible evaluation.
            this.RngStream = RandStream('mt19937ar', 'Seed', seed);
        end
    end
end
