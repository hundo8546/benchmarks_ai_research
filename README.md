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

root@929ad88a0cb4:/workspace/benchmarks_ai_research# python routing/evaluate_routing.py 
CNN predicted fake rate: 0.05
CNN-only accuracy: 0.545 cost: 1.0
Always escalate accuracy: 0.595 cost: 6.0
Bandit accuracy: 0.605 avg cost: 3.275
Qwen fake accuracy: 0.4731182795698925
Qwen real accuracy: 0.7009345794392523
Qwen predicted fake rate: 0.38
y_true
0    107
1     93
Name: count, dtype: int64
y_cnn
0    190
1     10
Name: count, dtype: int64
root@929ad88a0cb4:/workspace/benchmarks_ai_research# python routing/dcr.py 

===== DEPLOYMENT LATENCY REPORT =====
Mean Latency (ms): 83.83
P50 Latency (ms): 102.49
P95 Latency (ms): 139.54
Throughput (samples/sec): 11.93

Always Escalate Mean (ms): 416.23
Always Throughput (samples/sec): 2.4

Latency Savings: 0.7986
=====================================
