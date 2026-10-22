"""tab_biggan_resolution: BigGAN distant-shift subset (n=1500), verifier behaviour with the
images sent at native resolution vs. upscaled to 512 px longest edge (biggan_upscaled512.py).
Columns: accuracy, real recall, fake recall, predicted-fake rate, stale SigLIP2 accuracy, and a
paired-bootstrap 95% CI on (verifier - stale SigLIP2)."""
import os
import sys

import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = [("GPT-5.5", "gpt55", "y_verifier"), ("Qwen2.5-VL-7B", "qwen7b", "y_qwen")]


def compute_biggan_resolution():
    sys.path.insert(0, ROUTING_BASE)
    import baseline_utils as bu
    pool = pd.read_csv(os.path.join(ROUTING_BASE, "biggan_distant_shift_1500.csv"))
    stale_acc = float((pool.y_stale_sig == pool.y_true).mean())
    rows = []
    for name, tag, col in SPEC:
        for res, suffix in [("native", ""), ("512px", "_upscaled512")]:
            f = os.path.join(ROUTING_BASE, f"biggan_distant_shift_{tag}{suffix}_preds.csv")
            d = pool[["path", "y_true", "y_stale_sig"]].merge(pd.read_csv(f)[["path", col]], on="path", how="left")
            n_invalid = int(d[col].isna().sum())
            d = d.dropna(subset=[col])
            y, p = d.y_true.values, d[col].astype(int).values
            gap, lo, hi = bu.paired_bootstrap_diff_ci(y, p, d.y_stale_sig.values, n_boot=10000, seed=42)
            rows.append({"verifier": name, "resolution": res, "n": len(d), "n_invalid": n_invalid,
                         "acc": float((p == y).mean()),
                         "real_recall": float((p[y == 0] == 0).mean()),
                         "fake_recall": float((p[y == 1] == 1).mean()),
                         "pred_fake_rate": float((p == 1).mean()),
                         "stale_siglip2_acc": stale_acc,
                         "gap_vs_stale_siglip2": gap, "gap_ci_lo": lo, "gap_ci_hi": hi})
    return pd.DataFrame(rows)
