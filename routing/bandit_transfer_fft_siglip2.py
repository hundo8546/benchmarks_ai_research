"""
Cross-dataset bandit transfer matrix using the fair FFT + SigLIP2 + Qwen
bandit (see retrain_bandits_fft_siglip2.py), matching the backbone used by
the disagreement rule it's compared against (Table 15 vs 16 fairness fix).

For each (train, test) pair, train a bandit on `train`'s training-split rows
and evaluate on `test`'s val split. Diagonal entries are in-domain.

GenBuster has no Qwen coverage outside its own val set, so it cannot serve
as a `train` source with the standard protocol; its row reuses the
internal-split-trained bandit from retrain_bandits_fft_siglip2.py (fit on a
held-out subset of its own val set) — caveated in the output.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from retrain_bandits_fft_siglip2 import (
    BASE, FULL_BANDIT_FEATURES, DATASETS, load_dataset,
    compute_bandit_targets, train_bandit,
)

DATASET_NAMES = ["GenBuster", "SD14", "BigGAN"]


def fit_full_bandit(ds_name, df, val_paths):
    """Standard protocol: train on the non-val rows."""
    df_t = compute_bandit_targets(df)
    train_mask = ~df_t["path"].isin(val_paths)
    train_idx = df_t[train_mask].index.values
    X = df_t[FULL_BANDIT_FEATURES].values
    y = df_t["target"].values
    if len(np.unique(y[train_idx])) < 2:
        return None, None
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y[train_idx])
    return model, scaler


def eval_bandit(model, scaler, df, val_paths):
    val_df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    X_val = val_df[FULL_BANDIT_FEATURES].values
    Xs = scaler.transform(X_val)
    action = model.predict(Xs).astype(int)
    pred = np.where(action == 1, val_df["y_qwen"].values, val_df["y_fft"].values)
    acc = float(np.mean(pred == val_df["y_true"].values))
    return acc, float(np.mean(action))


def main():
    data = {}
    for ds_name, prefix, qwen_filename, val_paths_file, _ in DATASETS:
        print(f"Loading {ds_name}...")
        df, val_paths = load_dataset(ds_name, qwen_filename, val_paths_file)
        data[ds_name] = (df, val_paths)

    print("\nFitting in-train-domain bandits...")
    models = {}
    for ds_name in ["SD14", "BigGAN"]:
        df, val_paths = data[ds_name]
        model, scaler = fit_full_bandit(ds_name, df, val_paths)
        models[ds_name] = (model, scaler)

    # GenBuster: reuse the internal-split bandit (no train-split Qwen coverage)
    print("GenBuster: using internal-split bandit (no train-split Qwen coverage)")
    df_gb, val_paths_gb = data["GenBuster"]
    model_gb, scaler_gb, internal_split, _ = train_bandit(df_gb, val_paths_gb)
    models["GenBuster"] = (model_gb, scaler_gb)

    print("\n" + "=" * 70)
    print("Cross-dataset FFT+SigLIP2 bandit transfer accuracy")
    print("=" * 70)
    print(f"{'Train':<12} {'Test':<12} {'Accuracy':>10} {'EscRate':>10}")
    print("-" * 70)

    rows = []
    for train_name in DATASET_NAMES:
        model, scaler = models[train_name]
        for test_name in DATASET_NAMES:
            df_test, val_paths_test = data[test_name]
            acc, esc = eval_bandit(model, scaler, df_test, val_paths_test)
            tag = " *" if train_name == test_name else ""
            print(f"{train_name:<12} {test_name:<12} {acc:>10.3f} {esc:>10.3f}{tag}")
            rows.append({"train": train_name, "test": test_name,
                         "accuracy": acc, "escalation_rate": esc})

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "bandit_transfer_fft_siglip2.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print("NOTE: GenBuster row uses a bandit fit on an internal split of its own "
          "val set (no train-split Qwen coverage exists) -- caveat when reporting.")


if __name__ == "__main__":
    main()
