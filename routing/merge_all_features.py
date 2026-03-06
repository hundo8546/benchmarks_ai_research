import pandas as pd

BASE = "/workspace/benchmarks_ai_research/routing/"

# Load individual outputs
df_cnn = pd.read_csv(BASE + "bandit_dataset.csv")
df_cnn.head()
df_clip = pd.read_csv(BASE + "clip_preds.csv")        # must contain path, y_clip, latency_clip_ms
df_qwen = pd.read_csv(BASE + "qwen_preds.csv")       # must contain path, y_qwen, latency_qwen_ms

# Merge step by step
df = df_cnn.merge(
    df_clip[["path", "y_clip", "latency_clip_ms"]],
    on="path",
    how="left"
)

df = df.merge(
    df_qwen[["path", "y_qwen", "latency_qwen_ms"]],
    on="path",
    how="left"
)


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
df.to_csv(BASE + "merged_dataset.csv", index=False)

print("Saved merged_dataset.csv")