import pandas as pd
import numpy as np

# Load predictions
preds = pd.read_csv("/workspace/benchmarks_ai_research/routing/clip_preds.csv")

# Load dataset index
index = pd.read_csv("/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv")

# Merge labels + generator
df = preds.merge(index[["path", "label", "generator"]], on="path")
df = df.rename(columns={"label": "y_true"})

# Load validation indices from training
val_idx = np.load("/workspace/benchmarks_ai_research/routing/clip_val_idx.npy")

# Restrict to validation samples
df_val = df.iloc[val_idx].copy()

# Compute correctness
df_val["correct"] = (df_val["y_clip"] == df_val["y_true"]).astype(int)

print("\nCLIP Validation Accuracy Per Generator\n")

for gen in sorted(df_val["generator"].unique()):
    gen_df = df_val[df_val["generator"] == gen]
    acc = gen_df["correct"].mean()
    print(f"{gen:10s}  {acc:.3f}   n={len(gen_df)}")

print("\nOverall validation accuracy:", df_val["correct"].mean())