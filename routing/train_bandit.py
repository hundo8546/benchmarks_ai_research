import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

LAMBDA = 0.05

# Load merged dataset (must include:
# y_cnn, y_clip or y_qwen, y_true,
# latency_cnn_ms, latency_clip_ms, latency_qwen_ms)

df = pd.read_csv("bandit_dataset.csv")

# Choose escalation model
ESCALATION_COL = "y_clip"   # or "y_qwen"

# Total cost if stop (CNN + CLIP already paid)
df["cost_stop"] = df["latency_cnn_ms"] + df["latency_clip_ms"]

# Total cost if escalate
df["cost_esc"] = (
    df["latency_cnn_ms"] +
    df["latency_clip_ms"] +
    df["latency_qwen_ms"]
)

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

# Context features
X = df[["confidence", "margin1", "entropy1"]].values
y = df["target"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_val)

print("Bandit policy accuracy (decision match):", accuracy_score(y_val, pred))

joblib.dump(model, "bandit_model.pkl")
print("Saved bandit_model.pkl")