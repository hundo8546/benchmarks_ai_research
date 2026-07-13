"""
Train (or load cached) in-domain SigLIP2 logistic-regression probes for
GenBuster / SD14 / BigGAN, using the same train/val split convention as
experiments/e0_fft_variant.py and experiments/e8_modern_diffusion.py
(extractors.common.load_val_paths, i.e. the {prefix}bandit_val_paths.npy
files), so predictions here are directly comparable to FFT-only and other
E0-E8 results.

Produces, per dataset:
  {prefix}siglip2_probe.pkl   -- (clf, scaler), reusable by other scripts
  {prefix}siglip2_preds.csv   -- path, y_siglip2, siglip2_proba  (val split only)

This mirrors clip_classifier.pkl / clip_preds.csv so downstream scripts
(cross_family_deployment.py, retrain_bandits.py) can swap CLIP for SigLIP2
without needing the missing sd14/biggan raw CLIP features.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extractors"))
from extractors.common import DATASETS, ROUTING_BASE, load_val_paths


def train_and_predict(ds_name):
    cfg = DATASETS[ds_name]
    prefix = cfg["prefix"]
    feat_file = os.path.join(ROUTING_BASE, f"{prefix}siglip2_features.csv")
    if not os.path.exists(feat_file):
        print(f"  {ds_name}: missing {feat_file}, skipping")
        return

    df = pd.read_csv(feat_file)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)

    val_paths = load_val_paths(cfg, df["path"].tolist())
    val_mask = df["path"].isin(val_paths)
    train_mask = ~val_mask

    probe_pkl = os.path.join(ROUTING_BASE, f"{prefix}siglip2_probe.pkl")
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
    joblib.dump((clf, scaler), probe_pkl)

    X_val = scaler.transform(X[val_mask])
    y_val = y[val_mask]
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    print(f"  {ds_name}: n_train={train_mask.sum()} n_val={val_mask.sum()} "
          f"SigLIP2 val accuracy={acc:.4f}  -> saved {probe_pkl}")

    preds_csv = os.path.join(ROUTING_BASE, f"{prefix}siglip2_preds.csv")
    out = pd.DataFrame({
        "path": df.loc[val_mask, "path"].values,
        "y_siglip2": y_pred,
        "siglip2_proba": y_prob,
    })
    out.to_csv(preds_csv, index=False)
    print(f"  Saved {preds_csv}")


def main():
    for ds_name in DATASETS:
        print(f"\n=== {ds_name} ===")
        train_and_predict(ds_name)


if __name__ == "__main__":
    main()
