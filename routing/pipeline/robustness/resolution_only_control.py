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

Extension (remainingupdates1.md Step 1): the same control for SDXL and FLUX,
each under its three real-image-source variants (ImageNet/genimage, COCO
val2017, PASCAL VOC original) -- the deployment-shift test sets used for
Table e8_results / CONSOLIDATED_full_sdxl_flux.csv. SDXL/FLUX are zero-shot
deployment test sets with no canonical train/eval split (see pipeline/
splits.py's CANONICAL_DATASETS docstring), so this control uses its own
reproducible stratified 70/30 train/eval split (seed=42) instead of a
canonical *_bandit_val_paths.npy file. Reads native width/height from
sdxl_flux_image_inventory.csv (built by sdxl_flux_image_inventory.py).

Output: routing/results/resolution_only_control.csv
  columns: dataset, accuracy, real_recall, fake_recall, n_eval,
           native_resolution_summary
Output: routing/results/resolution_only_control_sdxl_flux.csv
  columns: dataset, real_source, accuracy, real_recall, fake_recall, n_eval,
           native_resolution_summary
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extractors.common import DATASETS, RESULTS_DIR, ensure_results_dir
from pipeline.splits import get_train_mask, get_eval_mask
from pipeline.robustness.native_resolution import (
    build_native_resolution_table, summarize_resolutions,
)
from pipeline.robustness.sdxl_flux_image_inventory import build_inventory, DATASETS as SDXL_FLUX_SOURCES


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


def run_sdxl_flux_variant(ds_name, real_source, inv_df):
    df = inv_df[(inv_df.dataset == ds_name) & (inv_df.real_source == real_source)].reset_index(drop=True)
    X = make_resolution_features(df)
    y = df["y_true"].values.astype(int)

    train_idx, eval_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_train, y[train_idx])

    X_eval = scaler.transform(X[eval_idx])
    y_eval = y[eval_idx]
    y_pred = clf.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    real_mask = y_eval == 0
    fake_mask = y_eval == 1
    real_recall = float(np.mean(y_pred[real_mask] == 0)) if real_mask.sum() else float("nan")
    fake_recall = float(np.mean(y_pred[fake_mask] == 1)) if fake_mask.sum() else float("nan")

    summary = summarize_resolutions(df.iloc[eval_idx].reset_index(drop=True))

    label = f"{ds_name}/{real_source}"
    print(f"{label}: n_train={len(train_idx)} n_eval={len(eval_idx)} "
          f"acc={acc:.4f} real_recall={real_recall:.4f} fake_recall={fake_recall:.4f}")
    print(f"  eval-set native resolutions: {summary}")

    return {
        "dataset": ds_name, "real_source": real_source,
        "accuracy": acc, "real_recall": real_recall, "fake_recall": fake_recall,
        "n_eval": len(eval_idx), "native_resolution_summary": summary,
    }


def run_sdxl_flux():
    inv_df = build_inventory()
    rows = [run_sdxl_flux_variant(ds_name, real_source, inv_df)
            for ds_name in SDXL_FLUX_SOURCES for real_source in SDXL_FLUX_SOURCES[ds_name]]
    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "resolution_only_control_sdxl_flux.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(out.to_string(index=False))
    return out


def main():
    ensure_results_dir()
    rows = [run_dataset(ds) for ds in DATASETS]
    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "resolution_only_control.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(out.to_string(index=False))

    run_sdxl_flux()


if __name__ == "__main__":
    main()
