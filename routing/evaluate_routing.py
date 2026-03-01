import pandas as pd
import numpy as np
import joblib

LAMBDA = 0.05
K1 = 1.0
K2 = 5.0

df_cnn = pd.read_csv("bandit_dataset.csv")
df_qwen = pd.read_csv("qwen_preds.csv")

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
print("CNN predicted fake rate:", np.mean(df["y_cnn"]))
# Load trained bandit
bandit = joblib.load("bandit_model.pkl")

# CNN-only
cnn_acc = np.mean(df["y_cnn"] == df["y_true"])
cnn_cost = K1

# Always escalate
always_acc = np.mean(df["y_qwen"] == df["y_true"])
always_cost = K1 + K2

# Bandit routing
correct = []
costs = []

for _, row in df.iterrows():
    context = np.array([[row["confidence"], row["margin1"], row["entropy1"]]])
    action = bandit.predict(context)[0]

    if action == 0:
        # stop
        pred = row["y_cnn"]
        cost = K1
    else:
        # escalate
        pred = row["y_qwen"]
        cost = K1 + K2

    correct.append(pred == row["y_true"])
    costs.append(cost)

bandit_acc = np.mean(correct)
bandit_cost = np.mean(costs)

print("CNN-only accuracy:", cnn_acc, "cost:", cnn_cost)
print("Always escalate accuracy:", always_acc, "cost:", always_cost)
print("Bandit accuracy:", bandit_acc, "avg cost:", bandit_cost)
print("Qwen fake accuracy:",
      np.mean(df[df["y_true"]==1]["y_qwen"] ==
              df[df["y_true"]==1]["y_true"]))
print("Qwen real accuracy:",
      np.mean(df[df["y_true"]==0]["y_qwen"] ==
              df[df["y_true"]==0]["y_true"]))
print("Qwen predicted fake rate:", np.mean(df["y_qwen"]==1))