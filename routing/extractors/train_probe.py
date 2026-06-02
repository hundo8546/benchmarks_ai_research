"""
Generic probe trainer — trains a logistic regression probe on any feature CSV.

Supports data-fraction experiments for E1 (limited-label regime) and E6.

Outputs:
  - <prefix>_probe_<fraction>.pkl   — (clf, scaler) joblib bundle
  - <prefix>_probe_<fraction>_val.npy — val path indices

Usage:
    python train_probe.py \\
        --features /path/to/features.csv \\
        --out_prefix /path/to/output_prefix \\
        --fractions 1.0 0.5 0.25 0.1 0.05 0.01
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Path to feature CSV")
    p.add_argument("--out_prefix", required=True, help="Output path prefix")
    p.add_argument("--fractions", nargs="+", type=float, default=[1.0],
                   help="Training set fractions to try")
    p.add_argument("--latency_col", default=None,
                   help="Latency column name to drop from features")
    p.add_argument("--C", type=float, default=0.1, help="LR regularization")
    p.add_argument("--val_fraction", type=float, default=0.3,
                   help="Fraction of data held out as val set (fixed across fractions)")
    return p.parse_args()


def load_features(feat_file, latency_col=None):
    df = pd.read_csv(feat_file)
    drop_cols = ["path", "y_true", "generator"]
    if latency_col and latency_col in df.columns:
        drop_cols.append(latency_col)
    # Drop any remaining non-feature columns
    for col in df.columns:
        if col.startswith("latency_"):
            drop_cols.append(col)
    drop_cols = list(set(drop_cols))
    feat_cols = [c for c in df.columns if c not in drop_cols and c.startswith("f")]
    return df, feat_cols


def main():
    args = parse_args()
    df, feat_cols = load_features(args.features, args.latency_col)

    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)
    groups = df["generator"].values if "generator" in df.columns else np.arange(len(y))

    # Fixed train/val split (stratified by label)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=args.val_fraction,
        random_state=42, stratify=y
    )

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    results = []
    for frac in args.fractions:
        if frac < 1.0:
            # Sub-sample from the training portion
            n_train = max(1, int(len(train_val_idx) * frac))
            rng = np.random.default_rng(42)
            # Stratified sub-sample
            y_tv = y[train_val_idx]
            pos_idx = train_val_idx[y_tv == 1]
            neg_idx = train_val_idx[y_tv == 0]
            n_pos = max(1, int(n_train / 2))
            n_neg = max(1, n_train - n_pos)
            sampled = np.concatenate([
                rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
                rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
            ])
        else:
            sampled = train_val_idx

        X_train = X[sampled]
        y_train = y[sampled]
        X_val = X[test_idx]
        y_val = y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=2000, C=args.C, random_state=42)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_val_s)
        y_prob = clf.predict_proba(X_val_s)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except Exception:
            auc = float("nan")

        print(f"  Fraction {frac:.2f}: n_train={len(sampled)}, "
              f"acc={acc:.3f}, bal_acc={bal_acc:.3f}, auroc={auc:.3f}")

        out_pkl = f"{args.out_prefix}_frac{frac:.2f}.pkl"
        joblib.dump((clf, scaler), out_pkl)

        val_paths = df.iloc[test_idx]["path"].values
        np.save(f"{args.out_prefix}_frac{frac:.2f}_val_paths.npy", val_paths)

        results.append({
            "fraction": frac,
            "n_train": len(sampled),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "auroc": auc,
            "model_path": out_pkl,
        })

    df_results = pd.DataFrame(results)
    results_csv = f"{args.out_prefix}_probe_results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nSaved probe results to {results_csv}")
    return df_results


if __name__ == "__main__":
    main()
