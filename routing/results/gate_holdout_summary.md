# Gate held-out validation: revised N-recommendation
Total configurations: 35
Configurations with disagreement pool < 40 samples (too small for a meaningful held-out half/half split): 3/35
              table config      verifier  n_disagree
verifier_comparison   SD14       GPT-5.5          23
verifier_comparison BigGAN Qwen2.5-VL-7B          14
verifier_comparison BigGAN       GPT-5.5          14

## Old claim (data-leaking: oracle computed from the SAME full disagreement pool the bootstrap resample is drawn from)
- N=50: binary gate reaches >=90% oracle-agreement in 21/35 configs
- N=100: binary gate reaches >=90% oracle-agreement in 24/35 configs
- N=200: binary gate reaches >=90% oracle-agreement in 28/35 configs

## Revised, genuinely held-out numbers (calibrate on one half, freeze, evaluate on the disjoint other half)
- split= N=50, naive (point-estimate) rule: reaches >=90% test-oracle decision-match in 17/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0670; mean regret = 0.0143
- split= N=50, significance-aware (CI) rule: reaches >=90% test-oracle decision-match in 18/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0031; mean regret = 0.0202
- split=N=100, naive (point-estimate) rule: reaches >=90% test-oracle decision-match in 20/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0545; mean regret = 0.0075
- split=N=100, significance-aware (CI) rule: reaches >=90% test-oracle decision-match in 21/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0021; mean regret = 0.0127
- split=N=200, naive (point-estimate) rule: reaches >=90% test-oracle decision-match in 26/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0345; mean regret = 0.0034
- split=N=200, significance-aware (CI) rule: reaches >=90% test-oracle decision-match in 24/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0030; mean regret = 0.0087
- split= half, naive (point-estimate) rule: reaches >=90% test-oracle decision-match in 29/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0224; mean regret = 0.0021
- split= half, significance-aware (CI) rule: reaches >=90% test-oracle decision-match in 26/32 configs (excl. too-small pools); mean harmful-escalation rate = 0.0049; mean regret = 0.0067

## Bottom line
The previous 'N>=100 reaches reliable agreement in 24/35 configs' claim (and the equivalent N=200 claim) was computed by comparing bootstrap resamples of a pool against an oracle derived from THAT SAME pool -- it measures resampling stability, not generalization. Under a genuine held-out split (calibrate on one disjoint half, freeze the decision, score on the other half against an independent test-half oracle), the comparable counts above are substantially different and should replace the old claim in the paper. See tab_gate_holdout.csv for full per-configuration detail (including harmful-escalation rate and regret for both the current point-estimate rule and the CI-based rule) and tab_agreement_path_audit.csv for the agreement-path correlated-error audit.
