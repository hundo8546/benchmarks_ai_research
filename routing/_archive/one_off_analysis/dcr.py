import pandas as pd
import numpy as np

BASE = "/workspace/benchmarks_ai_research/routing/"

df_cnn  = pd.read_csv(BASE + "bandit_dataset.csv")
df_clip_lat = pd.read_csv(BASE + "clip_features.csv")
df_qwen = pd.read_csv(BASE + "qwen_preds.csv")
df_clip_preds = pd.read_csv(BASE + "clip_preds.csv")
df_decisions = pd.read_csv(BASE + "bandit_decisions.csv")

val_paths = np.load(BASE + "bandit_val_paths.npy", allow_pickle=True)

# Build base df filtered to val paths
df = df_cnn.merge(df_qwen[["path", "latency_qwen_ms"]], on="path", how="left")
df = df.merge(df_clip_lat[["path", "latency_clip_ms"]], on="path", how="left")
df = df[df["path"].isin(val_paths)].reset_index(drop=True)

df["latency_qwen_ms"] = df["latency_qwen_ms"].fillna(0)

# Bandit escalation decisions
df = df.merge(df_decisions[["path", "escalate"]], on="path", how="left")
df["escalate"] = df["escalate"].fillna(0)

# Disagreement escalation
df_disagree = df_cnn.merge(df_clip_preds[["path", "y_clip"]], on="path")
df_disagree = df_disagree[df_disagree["path"].isin(val_paths)].reset_index(drop=True)
df_disagree["escalate_disagree"] = (df_disagree["y_cnn"] != df_disagree["y_clip"]).astype(int)
df = df.merge(df_disagree[["path", "escalate_disagree"]], on="path", how="left")
df["escalate_disagree"] = df["escalate_disagree"].fillna(0)

# Latencies
df["total_latency_ms"] = (
    df["latency_cnn_ms"]
    + df["latency_clip_ms"]
    + df["escalate"] * df["latency_qwen_ms"]
)
df["always_latency_ms"] = (
    df["latency_cnn_ms"]
    + df["latency_clip_ms"]
    + df["latency_qwen_ms"]
)
df["disagree_latency_ms"] = (
    df["latency_cnn_ms"]
    + df["latency_clip_ms"]
    + df["escalate_disagree"] * df["latency_qwen_ms"]
)

mean_latency       = df["total_latency_ms"].mean()
p50_latency        = np.percentile(df["total_latency_ms"], 50)
p95_latency        = np.percentile(df["total_latency_ms"], 95)
always_mean        = df["always_latency_ms"].mean()
disagree_mean      = df["disagree_latency_ms"].mean()

throughput         = 1000 / mean_latency
always_throughput  = 1000 / always_mean
disagree_throughput = 1000 / disagree_mean

latency_savings         = 1 - (mean_latency / always_mean)
disagree_latency_savings = 1 - (disagree_mean / always_mean)
entropy_latency = (df["latency_cnn_ms"] + df["latency_clip_ms"]).mean()
print("Entropy threshold Mean (ms):", round(entropy_latency, 2))

print("\n===== DEPLOYMENT LATENCY REPORT =====")
print("Mean Latency (ms):",        round(mean_latency, 2))
print("P50 Latency (ms):",         round(p50_latency, 2))
print("P95 Latency (ms):",         round(p95_latency, 2))
print("Throughput (samples/sec):", round(throughput, 2))
print()
print("Always Escalate Mean (ms):",       round(always_mean, 2))
print("Always Throughput (samples/sec):", round(always_throughput, 2))
print()
print("Bandit Latency Savings:",          round(latency_savings, 4))
print()
print("Disagree Mean (ms):",              round(disagree_mean, 2))
print("Disagree Throughput (samples/sec):", round(disagree_throughput, 2))
print("Disagree Latency Savings:",        round(disagree_latency_savings, 4))
print("Mean CNN latency (ms):", df["latency_cnn_ms"].mean().round(2))
print("Mean CLIP latency (ms):", df["latency_clip_ms"].mean().round(2))
print("Mean Qwen latency (ms):", df["latency_qwen_ms"][df["latency_qwen_ms"] > 0].mean().round(2))
print("=====================================")