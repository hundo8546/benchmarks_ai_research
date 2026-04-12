import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

LAMBDA = 0.05
K1 = 1.0
K2 = 5.0

# Load datasets
df_cnn = pd.read_csv("bandit_dataset.csv")
df_qwen = pd.read_csv("qwen_preds.csv")

# Merge on path
df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")

# Define escalation result
df["y_full"] = df["y_qwen"]

# Compute rewards
def compute_rewards(row):
    # Stop reward
    if row["y_cnn"] == row["y_true"]:
        r_stop = 1 - LAMBDA * K1
    else:
        r_stop = -1 - LAMBDA * K1

    # Escalate reward
    if row["y_full"] == row["y_true"]:
        r_esc = 1 - LAMBDA * (K1 + K2)
    else:
        r_esc = -1 - LAMBDA * (K1 + K2)

    return r_stop, r_esc

rewards = df.apply(compute_rewards, axis=1)
df["r_stop"] = [r[0] for r in rewards]
df["r_esc"] = [r[1] for r in rewards]

# Target = 1 if escalate better
df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)

X = df[["confidence", "margin1", "entropy1"]].values
y = df["target"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_val)

print("Bandit decision accuracy:", accuracy_score(y_val, pred))

joblib.dump(model, "bandit_model.pkl")
print("Saved bandit_model.pkl")