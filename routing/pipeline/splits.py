"""
routing.pipeline.splits — single source of truth for train/eval splitting.

Background (Phase 0 remediation, see routing/pipeline/SPLITS.md for full
narrative): three of four in-domain probes (FFT, SigLIP2, DINOv2) already use
the correct, simple scheme:

    val_paths  = load_val_paths(cfg, df["path"].tolist())   # {prefix}bandit_val_paths.npy
    val_mask   = df["path"].isin(val_paths)
    train_mask = ~val_mask

i.e. the eval set is the fixed, pre-drawn {prefix}bandit_val_paths.npy file
(GenBuster=1000, SD14=2000, BigGAN=2000 samples, written by
routing/expand_val_splits.py), and the training set is *everything else* in
the dataset's feature CSV — no nested holdout, no independent re-split.

CLIP (routing/train_clip_classifier.py) was the one holdout that used its own
independent train_test_split/GroupShuffleSplit scheme, which is what caused
the leakage documented in the Phase 0 audit. This module exists so every
future script (retrain_clip.py included) imports the *same* split logic
instead of re-deriving it.

This module does NOT regenerate or modify any *_bandit_val_paths.npy file —
those are treated as already-correct, fixed ground truth.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.common import DATASETS, ROUTING_BASE, RESULTS_DIR, load_val_paths, ensure_results_dir

# Datasets this module covers. (SDXL/FLUX are deployment-shift test sets
# evaluated zero-shot with SD14-trained probes, not split themselves — see
# SPLITS.md item on real-image sourcing, item C of the Phase 0 plan.)
CANONICAL_DATASETS = ["GenBuster", "SD14", "BigGAN"]


def get_canonical_eval_paths(dataset_name):
    """
    Return the canonical, fixed eval-set paths for a dataset, as a set of
    path strings.

    This is a thin, documented wrapper around the existing
    extractors.common.load_val_paths() / {prefix}bandit_val_paths.npy files.
    It does NOT regenerate those files -- they are already correct and are
    shared, unmodified, by the FFT/SigLIP2/DINOv2 probes.

    dataset_name: one of "GenBuster", "SD14", "BigGAN" (keys of
        extractors.common.DATASETS).
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[dataset_name]
    npy = cfg.get("val_paths", "")
    if not npy or not os.path.exists(npy):
        raise FileNotFoundError(
            f"Canonical eval split file not found for {dataset_name}: {npy!r}. "
            "This module intentionally does not regenerate it -- see "
            "routing/expand_val_splits.py if it is genuinely missing."
        )
    return set(np.load(npy, allow_pickle=True).tolist())


def get_train_mask(df, dataset_name, path_col="path"):
    """
    Return a boolean np.ndarray the same length as `df`, True for rows that
    belong to the TRAIN split (i.e. are NOT in the canonical eval set).

    This implements, for any dataframe with a `path_col` column, the exact
    `train_mask = ~val_mask` pattern already used by
    routing/train_siglip2_classifiers.py, routing/train_dinov2_classifiers.py,
    and routing/experiments/e0_fft_variant.py.
    """
    eval_paths = get_canonical_eval_paths(dataset_name)
    val_mask = df[path_col].isin(eval_paths).to_numpy()
    return ~val_mask


def get_eval_mask(df, dataset_name, path_col="path"):
    """Complement of get_train_mask; True for rows in the canonical eval set."""
    eval_paths = get_canonical_eval_paths(dataset_name)
    return df[path_col].isin(eval_paths).to_numpy()


def _label_col(df):
    return "y_true" if "y_true" in df.columns else "label"


def compute_split_manifest(write=True):
    """
    Compute the exact real split counts for GenBuster/SD14/BigGAN:
        total_pool, n_train, n_eval, n_eval_real, n_eval_fake

    Returns a pandas.DataFrame and (if write=True) also writes it to
    routing/results/splits_manifest.csv.
    """
    rows = []
    for ds_name in CANONICAL_DATASETS:
        cfg = DATASETS[ds_name]
        index_df = pd.read_csv(cfg["index"])
        label_col = _label_col(index_df)
        total_pool = len(index_df)

        eval_paths = get_canonical_eval_paths(ds_name)
        n_eval = len(eval_paths)
        n_train = total_pool - n_eval

        eval_mask = index_df["path"].isin(eval_paths)
        n_eval_matched = int(eval_mask.sum())
        n_eval_fake = int(index_df.loc[eval_mask, label_col].sum())
        n_eval_real = n_eval_matched - n_eval_fake

        rows.append({
            "dataset": ds_name,
            "total_pool": total_pool,
            "n_train": n_train,
            "n_eval": n_eval,
            "n_eval_matched_in_index": n_eval_matched,
            "n_eval_real": n_eval_real,
            "n_eval_fake": n_eval_fake,
        })
        print(f"{ds_name}: total_pool={total_pool}  n_train={n_train}  "
              f"n_eval={n_eval} (matched in index={n_eval_matched})  "
              f"n_eval_real={n_eval_real}  n_eval_fake={n_eval_fake}")

    manifest = pd.DataFrame(rows)
    if write:
        ensure_results_dir()
        out_path = os.path.join(RESULTS_DIR, "splits_manifest.csv")
        manifest.to_csv(out_path, index=False)
        print(f"\nSaved {out_path}")
    return manifest


if __name__ == "__main__":
    compute_split_manifest()
