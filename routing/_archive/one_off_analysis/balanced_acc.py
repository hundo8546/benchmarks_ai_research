import pandas as pd
import numpy as np
import joblib
BASE = "/workspace/benchmarks_ai_research/routing/"

def balanced_acc(correct, y_true):
    correct = np.array(correct)
    y_true = np.array(y_true)
    fake_idx = y_true == 1
    real_idx = y_true == 0
    fake_recall = correct[fake_idx].mean()
    real_recall = correct[real_idx].mean()
    return (fake_recall + real_recall) / 2

# Load GenBuster
import joblib
from sklearn.utils import resample

eps = 1e-12
K1, K2 = 1.0, 5.0

for prefix, vp_file, label in [
    ("", "bandit_val_paths.npy", "GenBuster"),
    ("sd14_", "sd14_bandit_val_paths.npy", "SD14"),
    ("biggan_", "biggan_bandit_val_paths.npy", "BigGAN"),
]:
    df_cnn  = pd.read_csv(BASE + f"{prefix}bandit_dataset.csv")
    df_qwen = pd.read_csv(BASE + f"{prefix}qwen_preds.csv")
    df_clip = pd.read_csv(BASE + f"{prefix}clip_preds.csv")
    df = df_cnn.merge(df_qwen[["path","y_qwen"]], on="path")
    df = df.merge(df_clip[["path","y_clip","clip_proba"]], on="path")
    val_paths = np.load(BASE + vp_file, allow_pickle=True)
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)

    methods = {
        "CNN-only":        (df["y_cnn"] == df["y_true"]).astype(int).tolist(),
        "Entropy thresh":  [(1 if (row["entropy1"] > 0.5 and row["y_qwen"] == row["y_true"]) or
                             (row["entropy1"] <= 0.5 and row["y_cnn"] == row["y_true"]) else 0)
                            for _, row in df.iterrows()],
        "Always escalate": (df["y_qwen"] == df["y_true"]).astype(int).tolist(),
        "Disagree thresh": [int((row["y_qwen"] if row["disagree"]==1 else row["y_cnn"]) == row["y_true"])
                            for _, row in df.iterrows()],
        "Disagree ensemble": [int((row["y_clip"] if row["disagree"]==1 else row["y_cnn"]) == row["y_true"])
                              for _, row in df.iterrows()],
        "CLIP-only":       (df["y_clip"] == df["y_true"]).astype(int).tolist(),
    }

    print(f"\n=== {label} Balanced Accuracy ===")
    print(f"{'Method':<25} {'Accuracy':>10} {'Bal Acc':>10}")
    print("-" * 48)
    for name, correct in methods.items():
        acc = np.mean(correct)
        bal = balanced_acc(correct, df["y_true"].values)
        print(f"{name:<25} {acc:>10.3f} {bal:>10.3f}")