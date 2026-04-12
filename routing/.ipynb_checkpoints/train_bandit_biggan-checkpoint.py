import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import joblib

LAMBDA = 0.05

# Load merged dataset (must include:
# y_cnn, y_clip or y_qwen, y_true,
# latency_cnn_ms, latency_clip_ms, latency_qwen_ms)

df = pd.read_csv("/workspace/benchmarks_ai_research/routing/biggan_merged_dataset.csv")



# Choose escalation model
ESCALATION_COL = "y_qwen"   # or "y_qwen"

df["cost_stop"] = (df["latency_cnn_ms"] + df["latency_clip_ms"])
df["cost_esc"] = (df["latency_cnn_ms"] + df["latency_clip_ms"] + df["latency_qwen_ms"])

max_cost = df["cost_esc"].max()

df["cost_stop"] = df["cost_stop"] / max_cost
df["cost_esc"] = df["cost_esc"] / max_cost

# Reward calculation
def compute_rewards(row):
    # Stop reward
    if row["y_cnn"] == row["y_true"]:
        r_stop = 1 - LAMBDA * row["cost_stop"]
    else:
        r_stop = -1 - LAMBDA * row["cost_stop"]

    # Escalate reward (uses escalation model)
    if row[ESCALATION_COL] == row["y_true"]:
        r_esc = 1 - LAMBDA * row["cost_esc"]
    else:
        r_esc = -1 - LAMBDA * row["cost_esc"]

    return r_stop, r_esc

rewards = df.apply(compute_rewards, axis=1)
df["r_stop"] = [r[0] for r in rewards]
df["r_esc"] = [r[1] for r in rewards]

# Target = 1 if escalate better
df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)
print(df["target"].value_counts())

# Context features
X = df[[
    "confidence", "margin1", "entropy1", "logit", "y_cnn",
    "y_clip", "clip_proba", "clip_margin", "clip_entropy",
    "disagree", "prob_gap"
]].values
y = df["target"].values

all_idx = np.arange(len(df))
train_idx, val_idx = train_test_split(all_idx, test_size=0.4, random_state=42)
val_paths = df.iloc[val_idx]["path"].values
np.save("/workspace/benchmarks_ai_research/routing/biggan_bandit_val_paths.npy", val_paths)



X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_val)

print("Bandit policy accuracy (decision match):", accuracy_score(y_val, pred))

print(classification_report(y_val, pred))
joblib.dump((model, scaler), "/workspace/benchmarks_ai_research/routing/biggan_bandit_model.pkl")

print("Saved bandit_model.pkl")