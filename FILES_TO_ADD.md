# Exact contents of the XRL-LLM repository (v1.0)

Build a NEW folder called `XRL-LLM` and copy in exactly what is listed below. Everything not listed stays out.

## Root (8 files)

```
README.md            new version from this pack
LICENSE              unchanged
CITATION.cff         new version from this pack
.gitignore           new version from this pack (blocks figures and manuscript files)
setupPaths.m
checkDependencies.m
main.m
runFullStudy.m
```

Do NOT copy the old `README.md` that sits in `XRL_LLM_Project` (the development notes). The new README replaces it.

## Code folders: copy whole folders, .m files only (74 files in total, including the 4 root .m files)

```
config/
    mergeConfigDefaults.m
    xrlConfig.m
core/
    applyScenario.m
    buildFeeder.m
    buildObservation.m
    checkAgentConfig.m
    computeInterpretabilityPenalty.m
    computeReward.m
    decodeAction.m
    describeAction.m
    featureMapping.m
    ieee33BusData.m
    ieee69BusData.m
    loadProfiles.m
    observationNormalization.m
    radialPowerFlow.m
    shapGroups.m
    solveWithDevices.m
    voltageSensitivity.m
    xrlReset.m
    xrlStep.m
environment/
    XRL33Env.m
training/
    createPPOAgent.m
    rewardWeightSweep.m
    trainPPO.m
policy/
    makePolicyFcn.m
    rulePolicy.m
shap/
    coalitionValues.m
    collectAttributions.m
    exactSHAP.m
    findTopFeatures.m
    kernelSHAP.m
    plotSHAPBar.m
    plotSHAPBeeswarm.m
    shapValueFunction.m
    summariseGroupValues.m
llm/
    connectLLM.m
    generateExplanation.m
    generateLayeredExplanations.m
    generateTemplateExplanation.m
    llmGenerate.m
    promptTemplates.m
    stripPreamble.m
evaluation/
    ablationExperiments.m
    ablationRepeats.m
    controlMetrics.m
    counterfactualTest.m
    diagnosePolicy.m
    gEval.m
    llmLatencyProfile.m
    loadBestCheckpoint.m
    rewardBalance.m
transfer/
    checkTransfer69.m
    runRobustness33.m
    runTransfer69.m
    xrlConfig69.m
dashboard/
    xrlDashboard.m
utils/
    plotTrainingProgress.m
    plotVoltageProfile.m
    saveRecords.m
tests/
    check.m
    checkNear.m
    report.m
    runAllTests.m
    testSuite.m
    test_environment.m
    test_llm.m
    test_powerFlow.m
    test_reward.m
    test_shap.m
```

Make sure `core/featureMapping.m` is the corrected version with the labels "(bus 2-17)" and "(bus 18-33)".

## figures/ (3 scripts, NO images)

```
figures/
    make_fig4_profile.m
    make_fig9_dashboard.m
    relabel_load_groups.m
```

## results/ (numerical results only, NO images)

```
results/
    trained_ppo_agent.mat            33-bus agent used in the paper
    xrl_records.mat                  explained decisions (Tables II-IV)
    xrl_records.json
    robustness33.mat                 Table IX rows C0-C8, attribution stability
    robustness33_summary.txt
    phase_explain_<timestamp>.mat    latest one only: three-level explanations (used by Fig. 9)
    ieee69/
        trained_ppo_agent_69.mat
        xrl69_records.mat
        xrl69_records.json
        transfer69.mat
        transfer69_summary.txt
```

Optional, if each is under 50 MB: the other `phase_*.mat` files (repeats, modelsize, sweep), which hold Tables V-VII. Check sizes in File Explorer first: GitHub rejects files over 100 MB.

## logs/ (plain text console output, NOT the supplementary PDFs)

```
logs/
    runAllTests.txt                  paste of the 115/115 output
    checkTransfer69.txt              paste of the 14 PASS lines
    full_study_console.txt           MATLAB command-window text of the full run
```

To create these, copy the text from the MATLAB Command Window into Notepad and save as .txt. Don't upload the formatted Supplementary Material PDFs: once they are submitted to IEEE, they fall under the same copyright terms as the paper.

## Leave out

- every .png / .tif / .pdf / .fig / .eps image, including those in results/ (the .gitignore blocks them anyway)
- the manuscript (.docx, .tex), the 2-page digest, draw.io sources, supplementary PDFs
- checkpoints/ and savedAgents/ folders
- anything containing passwords, API keys or reviewer correspondence

## Final check before pushing

In MATLAB, from inside `XRL-LLM`:
```matlab
setupPaths; runAllTests; checkTransfer69
```
Both must pass. Then in GitHub Desktop, look through the "Changes" list once. It should show only .m, .mat, .json, .txt, .md, .cff, LICENSE and .gitignore files: no images, no Word files.
