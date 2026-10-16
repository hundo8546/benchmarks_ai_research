"""
routing.pipeline.tables.per_generator — tab_per_generator.

*** PROVENANCE NOTE -- READ BEFORE TOUCHING THIS FILE ***

currentpaper.tex's Table~\\ref{tab:per_generator} ("Per-generator detection
accuracy on GenBuster-200K-mini") is a DESCRIPTIVE, intentionally SMALL
N=400 GenBuster subset: 73 CogVideoX + 73 EasyAnimate + 16 HunyuanVideo +
33 LTX-Video + 205 Real = 400. This predates the later eval-set expansion to
N=1000 (routing/expand_val_splits.py / routing/README.md).

The script that originally produced it, _archive/one_off_analysis/
escalation_table.py (imports from analysis2.py), is now archived. Critically,
analysis2.py itself loads the CURRENT bandit_val_paths.npy (N=1000) at module
level -- so simply re-running the archived script today would silently
reproduce this table at N=1000 instead of the original N=400. That is NOT a
bug to "fix" to N=1000: the paper explicitly describes this as a separate,
smaller, descriptive table, so N=400 is the CORRECT target here.

This module makes that provenance explicit and reproducible instead of
implicit and fragile: it loads bandit_val_paths_old_small.npy (the preserved
N=400 split) directly, by name, rather than going through
pipeline.splits.get_canonical_eval_paths() (which intentionally only knows
about the current, canonical N=1000/2000 splits and would raise/point at the
wrong file if asked for this one). Every row of the output is labeled with
split_provenance="descriptive_400_subset_old_small" so this is self-evident
from the CSV alone, not just from this docstring.

Inputs used (all already on disk, no new computation):
  bandit_dataset.csv          -- y_cnn (CNNSpot), y_true, generator (per-row
                                  fake-video generator tag, or "real")
  clip_preds.csv.leaky_backup -- y_clip predictions covering the OLD N=400
                                  split. NOTE: the CURRENT clip_preds.csv (see
                                  pipeline/retrain_clip.py) only covers the
                                  NEW N=1000 eval split -- it does not cover
                                  bandit_val_paths_old_small.npy at all, so it
                                  cannot be used for this archival table. The
                                  "*.leaky_backup" CLIP classifier predates
                                  the Phase 0 leak fix and is used here ONLY
                                  for path coverage of the old split, which is
                                  the right call for reproducing an archival,
                                  already-published descriptive table -- NOT
                                  appropriate for any live/primary result.
  qwen_preds.csv               -- y_qwen, current (v2, logit-decoded) Qwen
                                  predictions -- the best currently-available
                                  Qwen predictions for these paths. Because
                                  Qwen predictions have been regenerated since
                                  the paper was originally drafted (the v2
                                  logit-extraction fix), the Cascade column
                                  here may not bit-for-bit match the exact
                                  decimals already printed in the paper draft
                                  even though n and the CNNSpot/CLIP columns
                                  match exactly -- see this module's own
                                  self-test / the task report for the
                                  measured values.
"""
import os

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD_SMALL_VAL_PATHS = os.path.join(ROUTING_BASE, "bandit_val_paths_old_small.npy")

GENERATOR_ORDER = ["cogvideox", "easyanimate", "hunyuanvideo", "ltxvideo", "real"]
GENERATOR_LABEL = {
    "cogvideox": "CogVideoX", "easyanimate": "EasyAnimate",
    "hunyuanvideo": "HunyuanVideo", "ltxvideo": "LTX-Video", "real": "Real",
}


def compute_per_generator():
    """tab_per_generator: CNNSpot / Cascade(CNN+CLIP disagreement -> Qwen) /
    CLIP accuracy broken down by GenBuster generator, on the archival N=400
    subset (bandit_val_paths_old_small.npy), NOT the canonical N=1000 split."""
    if not os.path.exists(OLD_SMALL_VAL_PATHS):
        raise FileNotFoundError(
            f"{OLD_SMALL_VAL_PATHS} not found -- this is the preserved N=400 "
            "descriptive split this table intentionally depends on; see "
            "this module's docstring. Not falling back to the N=1000 split.")
    old_small_paths = set(np.load(OLD_SMALL_VAL_PATHS, allow_pickle=True).tolist())

    df_cnn = pd.read_csv(os.path.join(ROUTING_BASE, "bandit_dataset.csv"))
    clip_backup = os.path.join(ROUTING_BASE, "clip_preds.csv.leaky_backup")
    df_clip = pd.read_csv(clip_backup)
    df_qwen = pd.read_csv(os.path.join(ROUTING_BASE, "qwen_preds.csv"))

    df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path", how="inner")
    df = df.merge(df_clip[["path", "y_clip"]], on="path", how="inner")
    df = df[df["path"].isin(old_small_paths)].reset_index(drop=True)

    n_expected = len(old_small_paths)
    n_matched = len(df)
    coverage_note = (
        "full_coverage" if n_matched == n_expected
        else f"partial_coverage ({n_matched}/{n_expected} old-split paths "
             f"found in bandit_dataset.csv/clip_preds.csv.leaky_backup/qwen_preds.csv)"
    )

    disagree = (df["y_cnn"] != df["y_clip"])
    cascade_pred = np.where(disagree, df["y_qwen"], df["y_cnn"])
    df["cnn_correct"] = (df["y_cnn"] == df["y_true"]).astype(int)
    df["clip_correct"] = (df["y_clip"] == df["y_true"]).astype(int)
    df["cascade_correct"] = (cascade_pred == df["y_true"]).astype(int)

    rows = []
    for gen in GENERATOR_ORDER:
        mask = df["generator"] == gen
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "generator": GENERATOR_LABEL[gen],
            "n": n,
            "cnnspot_acc": float(df.loc[mask, "cnn_correct"].mean()),
            "cascade_acc": float(df.loc[mask, "cascade_correct"].mean()),
            "clip_acc": float(df.loc[mask, "clip_correct"].mean()),
            "split_provenance": "descriptive_400_subset_old_small",
            "coverage": coverage_note,
        })
    return pd.DataFrame(rows)
