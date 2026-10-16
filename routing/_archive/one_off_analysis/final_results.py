import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ----------------------------
# Cost settings
# ----------------------------
K1 = 1.0        # CNN cost
K2 = 5.0        # Qwen extra cost
LAMBDA = 0.05   # Not used for reporting, just reference

# ----------------------------
# Load datasets
# ----------------------------

df_cnn = pd.read_csv("/workspace/benchmarks_ai_research/routing/bandit_dataset.csv")
df_qwen = pd.read_csv("/workspace/benchmarks_ai_research/routing/qwen_preds.csv")
df_clip = pd.read_csv("/workspace/benchmarks_ai_research/routing/clip_preds.csv")

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
df = df.merge(df_clip, on="path")

df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)

# ----------------------------
# Baselines
# ----------------------------
cnn_acc = np.mean(df["y_cnn"] == df["y_true"])
clip_acc = np.mean(df["y_clip"] == df["y_true"])
qwen_acc = np.mean(df["y_qwen"] == df["y_true"])

# ----------------------------
# Train Bandit (MLP)
# ----------------------------
X = df[[
    "logit",
    "confidence",
    "margin1",
    "entropy1",
    "disagree"
]].values

# Target: escalate if Qwen fixes CNN
target = (
    (df["y_qwen"] == df["y_true"]) &
    (df["y_cnn"] != df["y_true"])
).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

bandit = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    max_iter=500,
    random_state=42
)
bandit.fit(X_scaled, target)

# ----------------------------
# Evaluate Routing
# ----------------------------
correct = []
costs = []
escalations = 0

for i, row in df.iterrows():
    context = np.array([X[i]])
    context = scaler.transform(context)
    action = bandit.predict(context)[0]

    if action == 1:
        # escalate
        pred = row["y_qwen"]
        cost = K1 + K2
        escalations += 1
    else:
        pred = row["y_cnn"]
        cost = K1

    correct.append(pred == row["y_true"])
    costs.append(cost)

bandit_acc = np.mean(correct)
bandit_cost = np.mean(costs)
escalation_rate = escalations / len(df)

# ----------------------------
# Compute savings
# ----------------------------
always_cost = K1 + K2
compute_savings = 1 - (bandit_cost / always_cost)

# ----------------------------
# Print Results
# ----------------------------
print("\n===== FINAL RESULTS =====")
print("CNN-only accuracy:        ", round(cnn_acc, 4))
print("CLIP-only accuracy:       ", round(clip_acc, 4))
print("Qwen-only accuracy:       ", round(qwen_acc, 4))
print("Always-escalate cost:     ", always_cost)
print()
print("Bandit accuracy:          ", round(bandit_acc, 4))
print("Bandit avg cost:          ", round(bandit_cost, 4))
print("Escalation rate:          ", round(escalation_rate, 4))
print("Compute savings vs always:", round(compute_savings, 4))
print()
print("Accuracy gain vs CNN:     ", round(bandit_acc - cnn_acc, 4))
print("Accuracy gain vs Qwen:    ", round(bandit_acc - qwen_acc, 4))