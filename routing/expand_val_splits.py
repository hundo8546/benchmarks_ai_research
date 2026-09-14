"""
Reviewer 1 Comment 1 — expand in-domain evaluation sets to >=2,000 samples
where the source pool allows (SD14, BigGAN both have a 12,000-image GenImage
val pool already fully feature-extracted). GenBuster's raw pool is only
2,000 samples total, so its ceiling is disclosed honestly rather than forced
to 2,000 val samples (would leave nothing to train the semantic/FFT probes
on).

Backs up existing *_bandit_val_paths.npy files (suffix _old_small.npy) and
writes new stratified (real/fake) val splits.
"""
import os
import shutil

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    "": {"csv": "bandit_dataset.csv", "n_val": 1000, "label": "GenBuster"},
    "sd14_": {"csv": "sd14_index.csv", "n_val": 2000, "label": "SD14"},
    "biggan_": {"csv": "biggan_index.csv", "n_val": 2000, "label": "BigGAN"},
}


def main():
    for prefix, cfg in TARGETS.items():
        label = cfg["label"]
        csv_path = os.path.join(ROUTING_BASE, cfg["csv"])
        df = pd.read_csv(csv_path)
        path_col = "path"
        label_col = "y_true" if "y_true" in df.columns else "label"

        n_total = len(df)
        n_val = min(cfg["n_val"], n_total // 2)  # never take more than half as val
        print(f"\n=== {label} ===")
        print(f"  Total pool: {n_total}  ->  requesting n_val={cfg['n_val']}, using n_val={n_val}")

        rng = np.random.default_rng(42)
        pos_idx = df.index[df[label_col] == 1].to_numpy().copy()
        neg_idx = df.index[df[label_col] == 0].to_numpy().copy()
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        n_pos_val = n_val // 2
        n_neg_val = n_val - n_pos_val
        n_pos_val = min(n_pos_val, len(pos_idx))
        n_neg_val = min(n_neg_val, len(neg_idx))

        val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
        val_paths = df.loc[val_idx, path_col].values

        val_npy_path = os.path.join(ROUTING_BASE, f"{prefix}bandit_val_paths.npy")
        backup_path = val_npy_path.replace(".npy", "_old_small.npy")
        if os.path.exists(val_npy_path) and not os.path.exists(backup_path):
            shutil.copy(val_npy_path, backup_path)
            print(f"  Backed up old val split -> {backup_path}")

        np.save(val_npy_path, val_paths)
        print(f"  Wrote new val split: n={len(val_paths)} "
              f"({n_pos_val} fake, {n_neg_val} real) -> {val_npy_path}")
        print(f"  New train pool: n={n_total - len(val_paths)}")


if __name__ == "__main__":
    main()
