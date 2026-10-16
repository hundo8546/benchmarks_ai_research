import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

BASE = "/workspace/benchmarks_ai_research/"

CNN_PATH = BASE + "bandit_dataset.csv"
QWEN_PATH = BASE + "qwen_preds.csv"
CLIP_PATH = BASE + "clip_preds.csv"

K1 = 1.0
K2 = 5.0
ALWAYS_COST = K1 + K2

# Load
df_cnn = pd.read_csv(CNN_PATH)
df_qwen = pd.read_csv(QWEN_PATH)
df_clip = pd.read_csv(CLIP_PATH)

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
df = df.merge(df_clip[["path", "y_clip"]], on="path")

df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)

# Features
X = df[[
    "logit",
    "confidence",
    "margin1",
    "entropy1",
    "flip",
    "disagree"
]].values

# Escalation target
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

# Get escalation probabilities
probs = router.predict_proba(X_scaled)[:, 1]

# Sweep thresholds
thresholds = np.linspace(0, 1, 50)

costs = []
accuracies = []

for t in thresholds:
    escalate = (probs > t).astype(int)

    preds = []
    sample_costs = []

    for i, row in df.iterrows():
        if escalate[i] == 1:
            preds.append(row["y_qwen"])
            sample_costs.append(K1 + K2)
        else:
            preds.append(row["y_cnn"])
            sample_costs.append(K1)

    acc = np.mean(np.array(preds) == df["y_true"].values)
    avg_cost = np.mean(sample_costs)

    accuracies.append(acc)
    costs.append(avg_cost)

# Baselines
cnn_acc = np.mean(df["y_cnn"] == df["y_true"])
qwen_acc = np.mean(df["y_qwen"] == df["y_true"])

# Plot
plt.figure(figsize=(6,5))
plt.plot(costs, accuracies)
plt.scatter(K1, cnn_acc)
plt.scatter(ALWAYS_COST, qwen_acc)

plt.xlabel("Average Cost Per Sample")
plt.ylabel("Accuracy")
plt.title("Smooth Cost–Accuracy Curve")
plt.grid(True)

plt.savefig(BASE + "smooth_cost_accuracy_curve.png", dpi=300)
plt.close()

print("Saved smooth_cost_accuracy_curve.png")