"""
Compute EXACT stale-probe disagreement counts (FFT vs SigLIP2, FFT vs
DINOv2) for SDXL/FLUX when the "real" class is swapped from the current
ImageNet/SD1.4 pool to either:
  --source original : AIGIBench's bundled PASCAL VOC real photographs
  --source coco      : an independent real-image source (COCO val2017)

This is the purely-local/free step that determines EXACTLY how many new
GPT-5.5 API calls a reproduction of the original-benchmark or
independent-source evaluation would need (n_disagree per backbone x
dataset) -- no API calls are made here.

Reuses:
  - routing/experiments/e8_modern_diffusion.py's get_predictions_stale()
    (same stale SD14-trained probes it already applies zero-shot)
  - EXISTING fake-image feature rows already in routing/sdxl_*_features.csv
    / routing/flux_*_features.csv (filtered to label==1) -- NOT recomputed.
  - NEW real-image feature rows in routing/results/{sdxl,flux}_realonly_*.csv
    (original) or routing/results/{sdxl,flux}_coco_realonly_*.csv (coco),
    produced by extract_original_real_features.py.

Usage:
    python analyze_real_source_disagreement.py --source original
    python analyze_real_source_disagreement.py --source coco
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROUTING = "/workspace/benchmarks_ai_research/routing"
RESULTS = os.path.join(ROUTING, "results")
sys.path.insert(0, ROUTING)
sys.path.insert(0, os.path.join(ROUTING, "experiments"))
from experiments.e8_modern_diffusion import get_predictions_stale  # noqa: E402

STALE_PROBES = {
    "FFT":     os.path.join(ROUTING, "sd14_fft_probe.pkl"),
    "SigLIP2": os.path.join(ROUTING, "sd14_siglip2_fftvariant_probe.pkl"),
    "DINOv2":  os.path.join(ROUTING, "sd14_dinov2_fftvariant_probe.pkl"),
}

FEAT_KEY = {"FFT": "fft_features", "SigLIP2": "siglip2_features", "DINOv2": "dinov2_features"}

DATASETS = ["sdxl", "flux"]


def existing_fake_preds(ds, backbone):
    """Apply the stale probe to the EXISTING mixed feature file (old real +
    fake) and return only the fake (y_true==1) rows -- these are the
    already-computed fake-image predictions, reused unchanged."""
    feat_file = os.path.join(ROUTING, f"{ds}_{FEAT_KEY[backbone]}.csv")
    df = get_predictions_stale(feat_file, STALE_PROBES[backbone], label=f"{ds}/{backbone}/fake")
    return df[df["y_true"] == 1].reset_index(drop=True)


def new_real_preds(ds, backbone, source):
    tag = ds if source == "original" else f"{ds}_coco"
    feat_file = os.path.join(RESULTS, f"{tag}_realonly_{FEAT_KEY[backbone]}.csv")
    if not os.path.exists(feat_file):
        raise FileNotFoundError(feat_file)
    df = get_predictions_stale(feat_file, STALE_PROBES[backbone], label=f"{ds}/{backbone}/new-real")
    return df.reset_index(drop=True)


def analyze(ds, source):
    rows = []
    for sem in ["SigLIP2", "DINOv2"]:
        fft_fake = existing_fake_preds(ds, "FFT")
        fft_real = new_real_preds(ds, "FFT", source)
        sem_fake = existing_fake_preds(ds, sem)
        sem_real = new_real_preds(ds, sem, source)

        fft_all = pd.concat([fft_fake, fft_real], ignore_index=True)
        sem_all = pd.concat([sem_fake, sem_real], ignore_index=True)

        df = fft_all.rename(columns={"y_pred": "y_fft", "y_prob": "prob_fft"}).merge(
            sem_all.rename(columns={"y_pred": "y_sem", "y_prob": "prob_sem"})[["path", "y_sem", "prob_sem"]],
            on="path", how="inner")

        n_total = len(df)
        y_true = df["y_true"].values
        y_fft = df["y_fft"].values
        y_sem = df["y_sem"].values
        disagree = (y_fft != y_sem)
        n_dis = int(disagree.sum())

        # Real-image subset (y_true == 0) -- fraction classified FAKE by each stale detector
        real_mask = (y_true == 0)
        n_real = int(real_mask.sum())
        frac_fft_calls_real_fake = float(np.mean(y_fft[real_mask] == 1)) if n_real else float("nan")
        frac_sem_calls_real_fake = float(np.mean(y_sem[real_mask] == 1)) if n_real else float("nan")

        fft_acc = float(np.mean(y_fft == y_true))
        sem_acc = float(np.mean(y_sem == y_true))

        rows.append({
            "source": source, "dataset": ds, "semantic_backbone": sem,
            "n_total": n_total, "n_real": n_real, "n_fake": n_total - n_real,
            "n_disagree": n_dis, "disagree_rate": n_dis / n_total,
            "fft_stale_acc": fft_acc, "sem_stale_acc": sem_acc,
            "frac_real_classified_fake_by_FFT": frac_fft_calls_real_fake,
            "frac_real_classified_fake_by_" + sem: frac_sem_calls_real_fake,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["original", "coco"], required=True)
    args = p.parse_args()

    all_rows = []
    for ds in DATASETS:
        all_rows.extend(analyze(ds, args.source))

    df = pd.DataFrame(all_rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    out_path = os.path.join(RESULTS, f"disagreement_{args.source}_real.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n=== EXACT n_disagree (= # new GPT-5.5 calls needed) ===")
    for _, r in df.iterrows():
        print(f"  {r['source']:8s} {r['dataset']:5s} {r['semantic_backbone']:8s}: "
              f"n_disagree={r['n_disagree']:4d}  (n_total={r['n_total']}, disagree_rate={r['disagree_rate']:.3f})")


if __name__ == "__main__":
    main()
