import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier

# =============================
# Paths
# =============================
BASE = "/workspace/benchmarks_ai_research/"

CNN_PATH = BASE + "bandit_dataset.csv"
QWEN_PATH = BASE + "qwen_preds.csv"
CLIP_PATH = BASE + "clip_preds.csv"

# =============================
# Cost Model
# =============================
K1 = 1.0   # CNN cost
K2 = 5.0   # Qwen extra cost
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

# =============================
# Baselines
# =============================
cnn_acc = np.mean(df["y_cnn"] == df["y_true"])
qwen_acc = np.mean(df["y_qwen"] == df["y_true"])
clip_acc = np.mean(df["y_clip"] == df["y_true"])

cnn_bal = balanced_accuracy_score(df["y_true"], df["y_cnn"])
qwen_bal = balanced_accuracy_score(df["y_true"], df["y_qwen"])

# =============================
# Train Router (same as final_results)
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

router = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    max_iter=500,
    random_state=42
)
router.fit(X_scaled, target)

# =============================
# Evaluate Routing
# =============================
correct = []
costs = []
actions = []

for i, row in df.iterrows():
    context = scaler.transform([X[i]])
    action = router.predict(context)[0]
    actions.append(action)

    if action == 1:
        pred = row["y_qwen"]
        cost = K1 + K2
    else:
        pred = row["y_cnn"]
        cost = K1

    correct.append(pred == row["y_true"])
    costs.append(cost)

bandit_acc = np.mean(correct)
bandit_bal = balanced_accuracy_score(df["y_true"],
                                     [df.iloc[i]["y_qwen"] if actions[i] == 1 else df.iloc[i]["y_cnn"]
                                      for i in range(len(df))])

bandit_cost = np.mean(costs)
escalation_rate = np.mean(actions)
compute_savings = 1 - (bandit_cost / ALWAYS_COST)

# =============================
# Print Summary
# =============================
print("\n===== FINAL METRICS =====")
print("CNN Accuracy:", round(cnn_acc,4))
print("CNN Balanced Acc:", round(cnn_bal,4))
print("Qwen Accuracy:", round(qwen_acc,4))
print("Qwen Balanced Acc:", round(qwen_bal,4))
print("CLIP Accuracy:", round(clip_acc,4))
print()
print("Bandit Accuracy:", round(bandit_acc,4))
print("Bandit Balanced Acc:", round(bandit_bal,4))
print("Bandit Avg Cost:", round(bandit_cost,4))
print("Escalation Rate:", round(escalation_rate,4))
print("Compute Savings:", round(compute_savings,4))

# =============================
# PLOT 1: Accuracy vs Cost
# =============================
plt.figure(figsize=(6,5))
plt.scatter(K1, cnn_acc)
plt.scatter(ALWAYS_COST, qwen_acc)
plt.scatter(bandit_cost, bandit_acc)

plt.xlabel("Average Cost Per Sample")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Compute Cost")
plt.grid(True)
plt.savefig(BASE + "accuracy_vs_cost.png", dpi=300)
plt.close()

# =============================
# PLOT 2: Cost Distribution
# =============================
plt.figure(figsize=(6,5))
plt.hist(costs, bins=2)
plt.xlabel("Per-Sample Cost")
plt.ylabel("Frequency")
plt.title("Cost Distribution (Stop vs Escalate)")
plt.grid(True)
plt.savefig(BASE + "cost_distribution.png", dpi=300)
plt.close()

# =============================
# PLOT 3: Escalation vs Savings
# =============================
plt.figure(figsize=(6,5))
plt.scatter(escalation_rate, compute_savings)
plt.xlabel("Escalation Rate")
plt.ylabel("Compute Savings")
plt.title("Escalation vs Compute Savings")
plt.grid(True)
plt.savefig(BASE + "escalation_vs_savings.png", dpi=300)
plt.close()

print("\nSaved figures:")
print(" - accuracy_vs_cost.png")
print(" - cost_distribution.png")
print(" - escalation_vs_savings.png")