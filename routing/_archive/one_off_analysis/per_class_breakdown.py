import pandas as pd
import numpy as np

BASE = "/workspace/benchmarks_ai_research/routing/"

DATASETS = {
    "GenBuster": BASE + "merged_dataset.csv",
    "SD14":      BASE + "sd14_merged_dataset.csv",
    "BigGAN":    BASE + "biggan_merged_dataset.csv",
}

def method_preds(df, method):
    if method == "CNN":
        return df["y_cnn"].values
    if method == "CLIP":
        return df["y_clip"].values
    if method == "Disagree":
        return np.where(df["disagree"] == 1, df["y_qwen"], df["y_cnn"])
    if method == "Ensemble":
        return np.where(df["disagree"] == 1, df["y_clip"], df["y_cnn"])
    raise ValueError(method)

METHODS = ["CNN", "CLIP", "Disagree", "Ensemble"]

for name, path in DATASETS.items():
    df = pd.read_csv(path)
    print(f"\n=== {name} ===")

    # Per-class (fake vs real)
    print("Per-class accuracy (fake recall / real recall):")
    header = f"{'Method':10s}"
    for m in METHODS:
        header += f"{m:>12s}"
    print(header)

    for label, label_name in [(1, "Fake"), (0, "Real")]:
        row = f"{label_name:10s}"
        mask = df["y_true"] == label
        for m in METHODS:
            preds = method_preds(df, m)
            acc = np.mean(preds[mask] == label) if mask.sum() > 0 else np.nan
            row += f"{acc:>12.3f}"
        print(row)

    # Per-generator (if generator column exists and has multiple values)
    if "generator" in df.columns and df["generator"].nunique() > 1:
        print("\nPer-generator accuracy:")
        header = f"{'Generator':15s}{'n':>6s}"
        for m in METHODS:
            header += f"{m:>12s}"
        print(header)
        for gen in sorted(df["generator"].unique()):
            mask = df["generator"] == gen
            n = mask.sum()
            row = f"{str(gen):15s}{n:>6d}"
            for m in METHODS:
                preds = method_preds(df, m)
                acc = np.mean(preds[mask] == df.loc[mask, "y_true"])
                row += f"{acc:>12.3f}"
            print(row)

    # Sanity check the mechanical equivalence
    clip_preds = method_preds(df, "CLIP")
    ens_preds = method_preds(df, "Ensemble")
    n_differ = np.sum(clip_preds != ens_preds)
    print(f"\nEnsemble vs CLIP: {n_differ} samples differ out of {len(df)}")