function cfg = xrlConfig69(varargin)
%XRLCONFIG69  Configuration for transferring XRL-LLM to the IEEE 69-bus feeder.
%
%   CFG = XRLCONFIG69() starts from XRLCONFIG (the 33-bus study settings)
%   and changes ONLY what the new feeder forces: the network data, device
%   placement, the loss reference used by the normalized reward, and the
%   output folder. Every PPO hyperparameter, reward weight, interpretability
%   setting, SHAP setting and prompt template is inherited unchanged. That
%   is the point of the experiment: it tests whether the METHOD transfers
%   without re-tuning, not whether a tuned 69-bus controller can be built.
%
%   CFG = XRLCONFIG69('name', value, ...) passes overrides to XRLCONFIG.
%
%   What has to change, and why
%   ---------------------------
%   * Observation size. 69 voltages + 68 loads + 3 PV + 3 context + 9
%     previous actions = 152 features (80 on the 33-bus feeder). The policy
%     network therefore cannot be reused; the agent is RETRAINED with the
%     same hyperparameters. The 12-player SHAP grouping is preserved
%     (IEEE69BUSDATA defines six spatial voltage bands and two load blocks),
%     so exact Shapley enumeration still costs 2^12 = 4096 coalitions.
%   * Device placement. Same device count and ratings as the 33-bus study
%     (OLTC + 5 x 300 kvar banks + 3 x 1200 kW PV), so the action space is
%     still 9-D. Banks sit on the main feeder (12, 21), the lateral off bus
%     4 (50) and the long lateral (61, 64) where the uncontrolled minimum
%     voltage occurs; PV sits at the main-feeder end (27), the end of the
%     lateral off bus 3 (35) and the long lateral (61). With 3.6 MW of PV
%     against 3.80 MW of load, low-load sunny days produce reverse power
%     flow and overvoltage at bus 27, mirroring the 33-bus scenario mix.
%   * Loss reference. cfg.reward.ref.Ploss is the uncontrolled mean loss
%     that normalises the loss term (80 kW on the 33-bus feeder). It is
%     scaled by the base-case loss ratio 224.99/202.7, giving ~89 kW.
%
%   See also XRLCONFIG, IEEE69BUSDATA, RUNTRANSFER69, CHECKTRANSFER69.

cfg = xrlConfig(varargin{:});

% --- Network -----------------------------------------------------------
cfg.net.dataFcn      = @ieee69BusData;
cfg.net.baseMVA      = 10.0;
cfg.net.baseKV       = 12.66;
cfg.net.totalLoadMW  = 3.8021;     % used by LOADPROFILES for the scenario tag
cfg.net.feederEndBus = 65;         % lowest-voltage bus of the uncontrolled feeder
cfg.net.description  = sprintf([ ...
    '  Network: IEEE 69-bus radial distribution feeder, %.2f kV, %.0f MVA base\n' ...
    '  Structure: one main feeder (buses 1-27) and laterals\n' ...
    '             (28-46 off bus 3, 47-50 off bus 4, 51-52 off bus 8,\n' ...
    '              53-65 off bus 9, 66-69 off buses 11-12)\n'], ...
    cfg.net.baseKV, cfg.net.baseMVA);

% --- Devices (same count and ratings as the 33-bus study) -------------
cfg.cap.buses  = [12 21 50 61 64];
cfg.cap.kvar   = [300 300 300 300 300];
cfg.pv.buses   = [27 35 61];
cfg.pv.ratedKW = [1200 1200 1200];

% --- Reward normalisation ---------------------------------------------
cfg.reward.ref.Ploss = 0.0089;     % ~89 kW (see header)

% --- Training length: identical to the 33-bus study -------------------
cfg.train.maxEpisodes = 6000;

% --- Keep every 69-bus artefact apart from the 33-bus results ---------
cfg.io.resultsDir   = fullfile('results', 'ieee69');
cfg.train.savePath  = fullfile('results', 'ieee69', 'trained_ppo_agent_69.mat');

% --- Re-derive dimensions (XRLCONFIG derived them for 33 buses) -------
d = cfg.net.dataFcn();
cfg.dims.nCap    = numel(cfg.cap.buses);
cfg.dims.nPV     = numel(cfg.pv.buses);
cfg.dims.nAction = 1 + cfg.dims.nCap + cfg.dims.nPV;
cfg.dims.nBus    = d.nBus;
cfg.dims.nObs    = d.nBus + (d.nBus - 1) + cfg.dims.nPV + 1 + 2 + cfg.dims.nAction;

assert(cfg.dims.nAction == 9,   'xrlConfig69:dims', 'Expected 9 action dims.');
assert(cfg.dims.nObs    == 152, 'xrlConfig69:dims', 'Expected 152 observation dims.');
end
