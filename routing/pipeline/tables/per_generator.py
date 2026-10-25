"""
routing.pipeline.tables.per_generator — tab_per_generator.

Per-generator accuracy on the canonical N=1,000 GenBuster evaluation set, using the same
per-sample predictions as tab_main_results (main_results.load_all): CNNSpot, FFT, SigLIP2,
and the primary FFT+SigLIP2 disagreement cascade with Qwen2.5-VL-7B. Generator tags come
from bandit_dataset.csv.

The earlier version of this table used an archival N=400 subset
(bandit_val_paths_old_small.npy), a CNNSpot+CLIP cascade, and the leaky pre-fix CLIP probe.
"""
import os

import numpy as np
import pandas as pd

from pipeline.tables import main_results

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENERATOR_ORDER = ["cogvideox", "easyanimate", "hunyuanvideo", "ltxvideo", "real"]
GENERATOR_LABEL = {
    "cogvideox": "CogVideoX", "easyanimate": "EasyAnimate",
    "hunyuanvideo": "HunyuanVideo", "ltxvideo": "LTX-Video", "real": "Real",
}


def compute_per_generator():
    df, _ = main_results.load_all("GenBuster")
    tags = pd.read_csv(os.path.join(ROUTING_BASE, "bandit_dataset.csv"))[["path", "generator"]]
    df = df.merge(tags, on="path", how="left")
    n_untagged = int(df["generator"].isna().sum())
    if n_untagged:
        raise ValueError(f"{n_untagged} GenBuster eval rows have no generator tag")

    disagree = df["y_fft"] != df["y_siglip2"]
    preds = {
        "cnnspot_acc": df["y_cnn"],
        "fft_acc": df["y_fft"],
        "siglip2_acc": df["y_siglip2"],
        "cascade_acc": pd.Series(np.where(disagree, df["y_qwen"], df["y_fft"]), index=df.index),
    }
    rows = []
    for gen in GENERATOR_ORDER:
        mask = df["generator"] == gen
        if not mask.any():
            continue
        row = {"generator": GENERATOR_LABEL[gen], "n": int(mask.sum()),
               "esc_rate": float(disagree[mask].mean())}
        for col, pred in preds.items():
            row[col] = float((pred[mask] == df.loc[mask, "y_true"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)
