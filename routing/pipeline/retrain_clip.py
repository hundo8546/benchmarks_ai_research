"""
routing.pipeline.retrain_clip — clean CLIP probe retraining for GenBuster.

Phase 0 remediation item D (fixes the CLIP-probe leakage, see SPLITS.md).

routing/train_clip_classifier.py built its OWN independent split
(train_test_split(test_size=0.5, stratify=y) -> clip_idx/bandit_idx, then a
GroupShuffleSplit(test_size=0.2) within clip_idx for probe validation), which
is NOT the same split as the later {prefix}bandit_val_paths.npy eval file
written by expand_val_splits.py. That meant ~40% of the current canonical
GenBuster eval set was actually inside clip_classifier.pkl's own training
data.

This script retrains CLIP for GenBuster using the exact same pattern already
used correctly by train_siglip2_classifiers.py / train_dinov2_classifiers.py
/ experiments/e0_fft_variant.py:

    val_paths  = load_val_paths(cfg, df["path"].tolist())
    val_mask   = df["path"].isin(val_paths)
    train_mask = ~val_mask

Scope: GenBuster only. SD14/BigGAN CLIP are explicitly NOT touched here —
their raw CLIP feature files no longer exist on disk, so this script cannot
(and does not try to) retrain or even verify them. See SPLITS.md.

Outputs (overwriting the leaky originals, which are backed up with a
`.leaky_backup` suffix before this script is first run):
    routing/clip_classifier.pkl   (clf, scaler) tuple, joblib-dumped
    routing/clip_preds.csv        path, y_clip, clip_proba   (eval split only)
"""
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.common import DATASETS, ROUTING_BASE

DATASET_NAME = "GenBuster"
FEATURES_CSV = os.path.join(ROUTING_BASE, "clip_features.csv")
CLASSIFIER_PKL = os.path.join(ROUTING_BASE, "clip_classifier.pkl")
PREDS_CSV = os.path.join(ROUTING_BASE, "clip_preds.csv")


def backup_if_needed():
    for path in (CLASSIFIER_PKL, PREDS_CSV):
        backup = path + ".leaky_backup"
        if os.path.exists(path) and not os.path.exists(backup):
            shutil.copy(path, backup)
            print(f"  Backed up {path} -> {backup}")


def retrain():
    cfg = DATASETS[DATASET_NAME]

    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)

    sys.path.insert(0, os.path.join(ROUTING_BASE))
    from pipeline.splits import get_train_mask, get_eval_mask
    train_mask = get_train_mask(df, DATASET_NAME)
    eval_mask = get_eval_mask(df, DATASET_NAME)

    backup_if_needed()

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
    joblib.dump((clf, scaler), CLASSIFIER_PKL)

    X_eval = scaler.transform(X[eval_mask])
    y_eval = y[eval_mask]
    y_pred = clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    acc = accuracy_score(y_eval, y_pred)

    print(f"{DATASET_NAME}: n_train={train_mask.sum()}  n_eval={eval_mask.sum()}  "
          f"NEW (clean) CLIP eval accuracy={acc:.4f}  -> saved {CLASSIFIER_PKL}")

    out = pd.DataFrame({
        "path": df.loc[eval_mask, "path"].values,
        "y_clip": y_pred,
        "clip_proba": y_prob,
    })
    out.to_csv(PREDS_CSV, index=False)
    print(f"  Saved {PREDS_CSV}")
    return acc


if __name__ == "__main__":
    retrain()
