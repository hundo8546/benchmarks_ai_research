import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

K1 = 1.0
K2 = 5.0
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

df_cnn = pd.read_csv("/workspace/benchmarks_ai_research/routing/bandit_dataset.csv")
df_qwen = pd.read_csv("/workspace/benchmarks_ai_research/routing/qwen_preds.csv")
df_clip = pd.read_csv("/workspace/benchmarks_ai_research/routing/clip_preds.csv")

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
df = df.merge(df_clip[["path", "y_clip", "latency_clip_ms"]], on="path")

val_paths = np.load("/workspace/benchmarks_ai_research/routing/bandit_val_paths.npy", allow_pickle=True)
df = df[df["path"].isin(val_paths)].reset_index(drop=True)


def evaluate_lambda(LAMBDA):
    df_local = df.copy()

    def compute_rewards(row):
        if row["y_cnn"] == row["y_true"]:
            r_stop = 1 - LAMBDA * K1
        else:
            r_stop = -1 - LAMBDA * K1
        if row["y_qwen"] == row["y_true"]:
            r_esc = 1 - LAMBDA * (K1 + K2)
        else:
            r_esc = -1 - LAMBDA * (K1 + K2)
        return r_stop, r_esc

    rewards = df_local.apply(compute_rewards, axis=1)
    df_local["r_stop"] = [r[0] for r in rewards]
    df_local["r_esc"] = [r[1] for r in rewards]
    df_local["target"] = (df_local["r_esc"] > df_local["r_stop"]).astype(int)

    X = df_local[["logit", "confidence", "margin1", "entropy1"]].values
    y = df_local["target"].values

    all_idx = np.arange(len(df_local))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])

    if len(np.unique(y[train_idx])) < 2:
        return np.mean(df_local["y_cnn"] == df_local["y_true"]), K1

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y[train_idx])

    correct = []
    costs = []
    for i in val_idx:
        row = df_local.iloc[i]
        context = scaler.transform([[row["logit"], row["confidence"], row["margin1"], row["entropy1"]]])
        action = model.predict(context)[0]
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        cost = K1 if action == 0 else K1 + K2
        correct.append(pred == row["y_true"])
        costs.append(cost)

    return np.mean(correct), np.mean(costs)


print("Lambda | Accuracy | Avg Cost")
print("--------------------------------")
for L in LAMBDAS:
    acc, cost = evaluate_lambda(L)
    print(f"{L:6} | {acc:.4f} | {cost:.3f}")

better = np.mean(
    (df["y_qwen"] == df["y_true"]) &
    (df["y_cnn"] != df["y_true"])
)
print("Fraction where escalation improves:", better)
print(np.corrcoef(df["confidence"], df["y_cnn"] == df["y_true"]))