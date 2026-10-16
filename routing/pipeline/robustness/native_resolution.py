"""
routing/pipeline/robustness/native_resolution.py

Reads NATIVE on-disk width/height for every sample in a dataset's full index
(before any resize), without decoding full pixel data:
  - images: PIL.Image.open(path).size (lazy header read, no pixel decode)
  - videos (GenBuster): cv2.VideoCapture frame-size metadata

This is the ground-truth input for the resolution-only confound control
(resolution_only_control.py) and the resolution-matched FFT ablation
(resolution_matched_ablation.py). Caches one CSV per dataset under
routing/results/ so repeated runs don't re-scan the whole pool.

Does NOT modify any file outside routing/results/ or this new directory.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extractors.common import DATASETS, RESULTS_DIR, load_index, ensure_results_dir

CACHE_DIR = RESULTS_DIR  # routing/results/


def _image_native_size(path):
    with Image.open(path) as im:
        return im.size  # (width, height)


def _video_native_size(path):
    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w, h)


def build_native_resolution_table(ds_name, force=False):
    """
    Returns a DataFrame with columns: path, y_true, generator, width, height
    for EVERY row in the dataset's full index (train + eval pool), reading
    native resolution from disk. Cached to
    routing/results/{ds_name}_native_resolution.csv.
    """
    ensure_results_dir()
    out_csv = os.path.join(CACHE_DIR, f"{ds_name}_native_resolution.csv")
    if os.path.exists(out_csv) and not force:
        return pd.read_csv(out_csv)

    cfg = DATASETS[ds_name]
    index_rows = load_index(cfg["index"])
    is_video = cfg["is_video"]

    rows_out = []
    for r in tqdm(index_rows, desc=f"native-res {ds_name}"):
        path = r["path"]
        try:
            if is_video:
                w, h = _video_native_size(path)
            else:
                w, h = _image_native_size(path)
        except Exception as e:
            print(f"  Skip {path}: {e}")
            continue
        rows_out.append({
            "path": path,
            "y_true": int(r["label"]),
            "generator": r.get("generator", "unknown"),
            "width": w,
            "height": h,
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows)")
    return df


def summarize_resolutions(df, label_col="y_true"):
    """Return a short human-readable summary string of native-resolution
    diversity, split by real (0) vs fake (1)."""
    parts = []
    for label, name in [(0, "real"), (1, "fake")]:
        sub = df[df[label_col] == label]
        if len(sub) == 0:
            continue
        res_strs = (sub["width"].astype(str) + "x" + sub["height"].astype(str))
        n_unique = res_strs.nunique()
        top = res_strs.value_counts()
        top_res, top_count = top.index[0], int(top.iloc[0])
        frac_top = top_count / len(sub)
        parts.append(
            f"{name}: n={len(sub)}, {n_unique} distinct native res, "
            f"top={top_res} ({frac_top:.1%})"
        )
    return " | ".join(parts)


if __name__ == "__main__":
    for ds in DATASETS:
        df = build_native_resolution_table(ds)
        print(ds, summarize_resolutions(df))
