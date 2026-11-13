# XRL-LLM: Explainable Reinforcement Learning with Natural-Language Justifications for Volt/VAR Control

[![DOI](https://zenodo.org/badge/DOI/ZENODO_DOI.svg)](https://doi.org/ZENODO_DOI)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2025b-orange.svg)](https://www.mathworks.com/products/matlab.html)

MATLAB code, trained agents and result records for the paper:

> E. E. Ogar, D. O. Ogar, E. Agabi, and S. Chaitusaney, "Explainable Reinforcement Learning with
> Natural-Language Justifications for Volt/VAR Control in Active Distribution Networks,"
> submitted to *IEEE Transactions on Industry Applications*, Special Issue on Transforming Power
> System Operations with Foundation Models, Explainable AI and Cyber-Resilience, 2026.

XRL-LLM couples three components into one pipeline:

1. an interpretability-aware PPO agent;
2. exact grouped Shapley attribution, using 12 semantic groups and 2^12 = 4096 coalitions;
3. domain-grounded LLM explanations at three levels (summary, standard, detailed).

The result is a natural-language justification for every Volt/VAR setpoint decision. The agent controls an OLTC, five capacitor banks and three PV inverters. It is evaluated against a tuned conventional VVC scheme and an uncontrolled feeder on the IEEE 33-bus feeder, and transferred to the IEEE 69-bus feeder.

This repository contains code and numerical results only. The paper's figures are not included. Every figure can be regenerated from the code and the archived records (see Section 3).

---

## 1. Requirements

| Item | Version / note |
|---|---|
| MATLAB | R2025b (tested: 25.2.0.3150157, Update 4). Run `checkDependencies` to verify your installation |
| Reinforcement Learning Toolbox | Required for training and for loading trained agents |
| Deep Learning Toolbox | Required for training and for loading trained agents |
| Statistics and Machine Learning Toolbox | Optional |
| [Ollama](https://ollama.com) | Local LLM server for the explanation and evaluation phases |
| LLM models | `ollama pull llama3:8b` (generator)<br>`ollama pull llama3.2:3b` (model-size ablation)<br>`ollama pull mistral-nemo` (G-Eval judge) |
| GPU | Optional. Results were produced on a 12 GB card (see the note in Section 3) |

Without the RL toolboxes, every stage after training still runs with the rule-based policy: power flow, attribution, explanations, evaluation and the dashboard. Without Ollama, explanations fall back to the domain-grounded template generator.

## 2. Repository layout

```
XRL-LLM/
  setupPaths.m          add all folders to the MATLAB path (call first)
  checkDependencies.m   report toolboxes and LLM availability
  runFullStudy.m        the study, in resource-safe phases (Section 3)
  main.m                single-command pipeline (quick checks / demos)
  config/               xrlConfig.m (all settings), mergeConfigDefaults.m
  core/                 feeders (ieee33BusData, ieee69BusData), power flow,
                        observation, reward, interpretability penalty, scenarios
  environment/          XRL33Env.m (RL Toolbox wrapper; feeder-generic)
  training/             createPPOAgent.m, trainPPO.m, rewardWeightSweep.m
  policy/               makePolicyFcn.m (deterministic batched policy), rulePolicy.m
  shap/                 exactSHAP, kernelSHAP, collectAttributions, plotting
  llm/                  prompt templates, Ollama client, template generator
  evaluation/           control metrics, counterfactual faithfulness, G-Eval,
                        ablations, latency profile, policy diagnostics
  transfer/             xrlConfig69, checkTransfer69, runTransfer69, runRobustness33
  dashboard/            xrlDashboard.m (interactive operator app)
  figures/              scripts that regenerate the paper's figures
  utils/                plotting and record export
  tests/                runAllTests.m and five test suites (115 checks)
  results/              trained agents and numerical result records (Section 5)
  logs/                 console output of the reported runs
```

## 3. Reproducing the paper: run order

Start MATLAB in the repository folder. All training uses seed 42, and all held-out evaluation uses seed 1042 (`cfg.seed + 1000`).

**Ollama and GPU memory:** stop Ollama before every training run. If MATLAB is holding GPU memory when Ollama loads a model, Ollama silently runs partly on the CPU. On a 12 GB card we measured a 58%/42% CPU/GPU split, which raised generation latency from 3–5 s to 16 s.

| # | Command | Ollama | Time* | Produces |
|---|---|---|---|---|
| 0 | `setupPaths; checkDependencies` | either | < 1 min | environment report |
| 1 | `runAllTests` | running | ~1 min | 115/115 checks |
| 2 | `runFullStudy('train')` | **stopped** | ~14 min | 33-bus agent, `results/trained_ppo_agent.mat` |
| 3 | `runFullStudy('explain')` | running | LLM-bound | Tables II–IV: control, regimes, faithfulness (fine and coarse grouping); explanations; G-Eval |
| 4 | `runFullStudy('repeats')` | running | overnight | Table V: ten-run paired ablation |
| 5 | `runFullStudy('modelsize')` | running | LLM-bound | Table VI: 8B vs 3B |
| 6 | `runFullStudy('sweep')` | **stopped** | overnight | Table VII: interpretability-weight trade-off |
| 7 | `checkTransfer69` | either | < 1 min | 14 checks: 33-bus pipeline unchanged; 69-bus base case reproduced |
| 8 | `R = runRobustness33();` | stopped | ~10 min | Table IX rows C0–C8; attribution stability |
| 9 | `T = runTransfer69();` | **stopped** | ~20 min | Table IX 69-bus row; 69-bus faithfulness |
| 10 | `make_fig4_profile` | either | < 1 min | voltage profile figure |
| 11 | `relabel_load_groups` | either | < 1 min | global Shapley bar and summary plots |
| 12 | `make_fig9_dashboard` | optional | ~1 min | operator dashboard figure |

\*Times are from the authors' workstation. LLM-bound phases scale with GPU memory and model residency.

Each phase saves its outputs to `results/` and can be re-run on its own. Later phases reload the trained agent automatically. Steps 10–12 read only the archived records in `results/`, so they work without retraining. Figure files are written to `results/` and are deliberately not version-controlled.

**Consistency check:** the first row (C0) of `runRobustness33` must reproduce Table II exactly: 0.320% / 0.013% violated bus-steps and 51.6 / 60.5 kW loss. If it doesn't, the evaluation scenarios differ from the paper's.

## 4. Key results (from the archived run)

| | IEEE 33-bus | IEEE 69-bus (transfer) |
|---|---|---|
| Loss vs conventional VVC | −14.7% | −32.1% |
| Switching vs conventional VVC | +88.1% | +85.7% |
| Violated bus-steps, XRL / conventional | 0.320% / 0.013% | 0.456% / 0.254% |
| Faithfulness, rank correlation (fine) | 0.306 ± 0.467 | 0.467 ± 0.294 |
| Sign agreement | 0.818 | 0.634 |
| Leading factor kept under 1% voltage noise | 90.1% | — |

## 5. What is archived and what is not

| Included | Not included |
|---|---|
| All source code and tests | Figures from the paper (regenerate them with Section 3, steps 10–12) |
| Trained agents: `results/trained_ppo_agent.mat`, `results/ieee69/trained_ppo_agent_69.mat` | The manuscript and its supplementary material |
| Attribution records and explanations: `results/xrl_records.mat` / `.json`, `results/ieee69/xrl69_records.mat` / `.json` | LLM weights (pull them with Ollama; see Section 1) |
| Robustness and transfer results: `results/robustness33.mat`, `results/ieee69/transfer69.mat` and their `*_summary.txt` files | Intermediate training checkpoints |
| Console logs of the reported runs (`logs/`) | MATLAB and toolbox installers |

**Note on SHAP group labels:** the archived records were produced with the load-group labels "Main-feeder load (bus 2-18)" and "Lateral load (bus 19-33)". The underlying observation indices (34:49 and 50:65) correspond to buses 2–17 and 18–33, and the current `core/featureMapping.m` uses those correct labels. Only the label text changes; all attribution values are unaffected. `figures/relabel_load_groups.m` applies the corrected labels when plotting from the archived records.

## 6. Data sources

- IEEE 33-bus: M. E. Baran and F. F. Wu, "Network reconfiguration in distribution systems for loss reduction and load balancing," *IEEE Trans. Power Del.*, vol. 4, no. 2, pp. 1401–1407, 1989.
- IEEE 69-bus: M. E. Baran and F. F. Wu, "Optimal capacitor placement on radial distribution systems," *IEEE Trans. Power Del.*, vol. 4, no. 1, pp. 725–734, 1989. Data transcribed from MATPOWER `case69.m`. Base case reproduced: 224.99 kW loss, minimum voltage 0.9092 p.u. at bus 65.
- Load and PV profiles are synthetic and generated analytically by `core/loadProfiles.m`, so no external data files are required.

## 7. Citation

If you use this code, please cite the paper and the archived software. GitHub's "Cite this repository" button, generated from `CITATION.cff`, gives the same information.

```bibtex
@software{ogar_xrl_llm_2026,
  author    = {Ogar, Emmanuel Ede and Ogar, Deborah Oyetolani and Agabi, Emmanuel and Chaitusaney, Surachai},
  title     = {{XRL-LLM: Explainable Reinforcement Learning with Natural-Language Justifications for Volt/VAR Control}},
  year      = {2026},
  version   = {v1.0},
  publisher = {Zenodo},
  doi       = {ZENODO_DOI},
  url       = {https://doi.org/ZENODO_DOI}
}
```

## 8. License and contact

This code is released under the MIT License (see [LICENSE](LICENSE)).

Contact: Emmanuel Ede Ogar, Department of Electrical Engineering, Chulalongkorn University (emmanueledeogar@gmail.com).
