import pandas as pd
import numpy as np

BASE = "/workspace/benchmarks_ai_research/routing/"

df_cnn = pd.read_csv(BASE + "bandit_dataset.csv")
df_clip = pd.read_csv(BASE + "clip_features.csv")   # if latency logged here
df_qwen = pd.read_csv(BASE + "qwen_preds.csv")

# Merge
df = df_cnn.merge(df_qwen[["path","latency_qwen_ms"]], on="path", how="left")
df = df.merge(df_clip[["path","latency_clip_ms"]], on="path", how="left")

# Assume CNN latency already in df_cnn as latency_cnn_ms

# Replace NaN for non-escalated samples
df["latency_qwen_ms"] = df["latency_qwen_ms"].fillna(0)

# Define escalation
df["escalate"] = df["escalate"] if "escalate" in df.columns else 0

# Compute per-sample total latency
df["total_latency_ms"] = (
    df["latency_cnn_ms"]
    + df["latency_clip_ms"]
    + df["escalate"] * df["latency_qwen_ms"]
)

# Always escalate baseline
df["always_latency_ms"] = (
    df["latency_cnn_ms"]
    + df["latency_clip_ms"]
    + df["latency_qwen_ms"]
)

# =============================
# Metrics
# =============================
mean_latency = df["total_latency_ms"].mean()
p50_latency = np.percentile(df["total_latency_ms"], 50)
p95_latency = np.percentile(df["total_latency_ms"], 95)

always_mean_latency = df["always_latency_ms"].mean()

throughput = 1000 / mean_latency
always_throughput = 1000 / always_mean_latency

latency_savings = 1 - (mean_latency / always_mean_latency)

print("\n===== DEPLOYMENT LATENCY REPORT =====")
print("Mean Latency (ms):", round(mean_latency,2))
print("P50 Latency (ms):", round(p50_latency,2))
print("P95 Latency (ms):", round(p95_latency,2))
print("Throughput (samples/sec):", round(throughput,2))
print()
print("Always Escalate Mean (ms):", round(always_mean_latency,2))
print("Always Throughput (samples/sec):", round(always_throughput,2))
print()
print("Latency Savings:", round(latency_savings,4))
print("=====================================")