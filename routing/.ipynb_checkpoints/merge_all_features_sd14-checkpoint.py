import pandas as pd
import numpy as np

BASE = "/workspace/benchmarks_ai_research/routing/"

# Load individual outputs
df_cnn  = pd.read_csv(BASE + "sd14_bandit_dataset.csv")
df_clip = pd.read_csv(BASE + "sd14_clip_preds.csv")
df_qwen = pd.read_csv(BASE + "sd14_qwen_preds.csv")
val_idx = np.load(BASE + "sd14_clip_val_idx.npy")
df_cnn  = df_cnn.iloc[val_idx].reset_index(drop=True)

# Merge step by step
df = df_cnn.merge(
    df_clip[["path", "y_clip", "clip_proba", "latency_clip_ms"]],
    on="path",
    how="left"
)

df = df.merge(
    df_qwen[["path", "y_qwen", "latency_qwen_ms"]],
    on="path",
    how="left"
)

#derived after merge
df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
df["clip_margin"] = np.abs(df["clip_proba"] - 0.5)
eps = 1e-12
df["clip_entropy"] = -(
    df["clip_proba"] * np.log(df["clip_proba"] + eps) +
    (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + eps)
)
df["prob_gap"] = np.abs(df["confidence"] - df["clip_proba"])



# Fill missing latency if some rows not escalated yet
df["latency_clip_ms"] = df["latency_clip_ms"].fillna(0)
df["latency_qwen_ms"] = df["latency_qwen_ms"].fillna(0)
df["latency_cnn_ms"] = df["latency_cnn_ms"].fillna(0)
# Sanity checks
print("Total rows:", len(df))
print("Missing clip preds:", df["y_clip"].isna().sum())
print("Missing qwen preds:", df["y_qwen"].isna().sum())
print("Missing cnn preds:", df["y_cnn"].isna().sum())

# Save
df.to_csv(BASE + "sd14_merged_dataset.csv", index=False)

print("Saved sd14_merged_dataset.csv")