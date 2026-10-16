"""
routing/pipeline/robustness/resolution_only_control.py

Resolution-only confound control (reviewer concern, peerreviewdecision2.txt
point 9): trains a trivial logistic-regression classifier on ONLY
[native_width, native_height, aspect_ratio, native_width*native_height]  --
no pixel content whatsoever -- to test whether the known resolution confound
(BigGAN/SD14 fakes are generated at a single fixed native resolution while
reals span 100+ native resolutions, see native_resolution.py) is, by itself,
sufficient to approximate the FFT detector's near-perfect accuracy.

Uses the SAME canonical train/eval split as every other probe in this repo
(routing.pipeline.splits.get_train_mask / get_canonical_eval_paths), applied
to the full native-resolution index built by native_resolution.py. No image
pixels are read for this control -- only on-disk width/height metadata.

GenBuster is included as a pre-registered sanity-check negative control:
it is video-frame-based and (per native_resolution.py's audit) both real and
fake GenBuster videos are uniformly 1024x1024, so resolution alone carries NO
class signal there -- this script should and does confirm that empirically.

Output: routing/results/resolution_only_control.csv
  columns: dataset, accuracy, real_recall, fake_recall, n_eval,
           native_resolution_summary
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extractors.common import DATASETS, RESULTS_DIR, ensure_results_dir
from pipeline.splits import get_train_mask, get_eval_mask
from pipeline.robustness.native_resolution import (
    build_native_resolution_table, summarize_resolutions,
)


def make_resolution_features(df):
    w = df["width"].values.astype(np.float64)
    h = df["height"].values.astype(np.float64)
    aspect = w / np.clip(h, 1, None)
    area = w * h
    return np.stack([w, h, aspect, area], axis=1)


def run_dataset(ds_name):
    df = build_native_resolution_table(ds_name)
    train_mask = get_train_mask(df, ds_name, path_col="path")
    eval_mask = get_eval_mask(df, ds_name, path_col="path")

    X = make_resolution_features(df)
    y = df["y_true"].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_train, y[train_mask])

    X_eval = scaler.transform(X[eval_mask])
    y_eval = y[eval_mask]
    y_pred = clf.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    real_mask = y_eval == 0
    fake_mask = y_eval == 1
    real_recall = float(np.mean(y_pred[real_mask] == 0)) if real_mask.sum() else float("nan")
    fake_recall = float(np.mean(y_pred[fake_mask] == 1)) if fake_mask.sum() else float("nan")

    summary = summarize_resolutions(df[eval_mask.astype(bool)].reset_index(drop=True))

    print(f"{ds_name}: n_train={train_mask.sum()} n_eval={eval_mask.sum()} "
          f"acc={acc:.4f} real_recall={real_recall:.4f} fake_recall={fake_recall:.4f}")
    print(f"  eval-set native resolutions: {summary}")

    return {
        "dataset": ds_name,
        "accuracy": acc,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "n_eval": int(eval_mask.sum()),
        "native_resolution_summary": summary,
    }


def main():
    ensure_results_dir()
    rows = [run_dataset(ds) for ds in DATASETS]
    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "resolution_only_control.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
