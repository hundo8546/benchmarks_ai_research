how to run:
run download_weights.py
(could run hfgenbuster)
run setup.sh

Pipeline:

Input frame
    ↓
generate_cnnspot_features.py
    ↓
generate_clip_features.py
    ↓
train_clip_classifier.py
    ↓
generate_clip_preds.py
    ↓
run_qwen_on_genbuster.py
    ↓
merge_all_features.py
    ↓
train_bandit.py
    ↓
evaluate_routing.py
    ↓
sweep_lambda.py
    ↓
deployment_evaluation.py


target
0    1063
1     937
Name: count, dtype: int64
Bandit policy accuracy (decision match): 0.5775
Saved bandit_model.pkl

CNN predicted fake rate: 0.052
CNN-only accuracy: 0.522 cost: 1.0
Always escalate accuracy: 0.603 cost: 6.0
Bandit accuracy: 0.5755 avg cost: 3.225
Qwen fake accuracy: 0.473
Qwen real accuracy: 0.733
Qwen predicted fake rate: 0.37

Lambda | Accuracy | Avg Cost
--------------------------------
   0.0 | 0.7195 | 2.783
  0.05 | 0.7195 | 2.783
   0.1 | 0.7195 | 2.783
   0.2 | 0.7195 | 2.783
   0.5 | 0.5220 | 1.000
   1.0 | 0.5220 | 1.000
Fraction where escalation improves: 0.234
[[1.        0.0458879]
 [0.0458879 1.       ]]
root@0e74093228f4:/workspace/benchmarks_ai_research# python routing/dcr.py          

===== DEPLOYMENT LATENCY REPORT =====
Mean Latency (ms): 18.48
P50 Latency (ms): 15.96
P95 Latency (ms): 37.47
Throughput (samples/sec): 54.1

Always Escalate Mean (ms): 366.78
Always Throughput (samples/sec): 2.73

Latency Savings: 0.9496
=====================================
root@0e74093228f4:/workspace/benchmarks_ai_research# python routing/final_results.py 

===== FINAL RESULTS =====
CNN-only accuracy:         0.522
CLIP-only accuracy:        0.988
Qwen-only accuracy:        0.603
Always-escalate cost:      6.0

Bandit accuracy:           0.719
Bandit avg cost:           2.7675
Escalation rate:           0.3535
Compute savings vs always: 0.5388

Accuracy gain vs CNN:      0.197
Accuracy gain vs Qwen:     0.116