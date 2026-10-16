import pandas as pd
import numpy as np
BASE = "/workspace/benchmarks_ai_research/routing/"
df = pd.read_csv(BASE + "merged_dataset.csv")
val_paths = np.load(BASE + "bandit_val_paths.npy", allow_pickle=True)
df = df[df["path"].isin(val_paths)].reset_index(drop=True)
df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
df["cnn_correct"] = (df["y_cnn"] == df["y_true"]).astype(int)
print(df.groupby(["cnn_correct", "disagree"]).size())