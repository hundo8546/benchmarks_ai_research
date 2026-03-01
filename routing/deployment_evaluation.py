import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

BASE = "/workspace/benchmarks_ai_research/"

CNN_PATH = BASE + "bandit_dataset.csv"
QWEN_PATH = BASE + "qwen_preds.csv"
CLIP_PATH = BASE + "clip_preds.csv"

# =============================
# Cost Model
# =============================
K1 = 1.0
K2 = 5.0
ALWAYS_COST = K1 + K2

# =============================
# Load Data
# =============================
df_cnn = pd.read_csv(CNN_PATH)
df_qwen = pd.read_csv(QWEN_PATH)
df_clip = pd.read_csv(CLIP_PATH)

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
df = df.merge(df_clip[["path", "y_clip"]], on="path")

df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
df["cnn_correct"] = (df["y_cnn"] == df["y_true"]).astype(int)

# =============================
# Train Router
# =============================
X = df[[
    "logit",
    "confidence",
    "margin1",
    "entropy1",
    "flip",
    "disagree"
]].values

target = (
    (df["y_qwen"] == df["y_true"]) &
    (df["y_cnn"] != df["y_true"])
).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

router = MLPClassifier(hidden_layer_sizes=(16, 8),
                       max_iter=500,
                       random_state=42)
router.fit(X_scaled, target)

probs = router.predict_proba(X_scaled)[:, 1]
threshold = 0.5
df["escalate"] = (probs > threshold).astype(int)

# =============================
# Final Predictions + Cost
# =============================
df["final_pred"] = np.where(df["escalate"] == 1,
                            df["y_qwen"],
                            df["y_cnn"])

df["cost"] = np.where(df["escalate"] == 1,
                      K1 + K2,
                      K1)

# =============================
# Accuracy Metrics
# =============================
accuracy = np.mean(df["final_pred"] == df["y_true"])

tn, fp, fn, tp = confusion_matrix(df["y_true"],
                                   df["final_pred"]).ravel()

tpr = tp / (tp + fn)   # Fake recall
tnr = tn / (tn + fp)   # Real recall
fpr = fp / (fp + tn)
balanced_acc = 0.5 * (tpr + tnr)

# =============================
# Escalation Targeting Analysis
# =============================
escalate_rate = np.mean(df["escalate"])

# Escalation when CNN correct vs wrong
esc_when_correct = np.mean(df[df["cnn_correct"] == 1]["escalate"])
esc_when_wrong = np.mean(df[df["cnn_correct"] == 0]["escalate"])

# Escalation when agree vs disagree
esc_when_agree = np.mean(df[df["disagree"] == 0]["escalate"])
esc_when_disagree = np.mean(df[df["disagree"] == 1]["escalate"])

# =============================
# Cost Statistics
# =============================
mean_cost = np.mean(df["cost"])
p50_cost = np.percentile(df["cost"], 50)
p95_cost = np.percentile(df["cost"], 95)
compute_savings = 1 - (mean_cost / ALWAYS_COST)

# =============================
# Print Report
# =============================
print("\n===== DEPLOYMENT EVALUATION REPORT =====")

print("\n--- Accuracy Metrics ---")
print("Overall Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))
print("Fake Recall (TPR):", round(tpr, 4))
print("Real Recall (TNR):", round(tnr, 4))
print("False Positive Rate:", round(fpr, 4))

print("\n--- Escalation Behavior ---")
print("Overall Escalation Rate:", round(escalate_rate, 4))
print("Escalate when CNN correct:", round(esc_when_correct, 4))
print("Escalate when CNN wrong:", round(esc_when_wrong, 4))
print("Escalate when Agree:", round(esc_when_agree, 4))
print("Escalate when Disagree:", round(esc_when_disagree, 4))

print("\n--- Compute Metrics ---")
print("Mean Cost:", round(mean_cost, 4))
print("P50 Cost:", round(p50_cost, 4))
print("P95 Cost:", round(p95_cost, 4))
print("Compute Savings vs Always Escalate:",
      round(compute_savings, 4))

print("\n========================================")