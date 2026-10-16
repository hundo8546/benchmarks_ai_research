"""
routing/pipeline/robustness/e7_compression_extended.py

Extends the original JPEG-compression robustness ablation
(routing/experiments/e7_compression.py, results/e7_compression.csv -- which
only covers CNNSpot/CLIP, on the OLD pre-expansion val splits of
n=400/480/480) to the FFT, SigLIP2 and DINOv2 detectors, evaluated on the
CURRENT canonical eval splits (routing.pipeline.splits.get_canonical_eval_paths
-- GenBuster n=1000, SD14 n=2000, BigGAN n=2000).

For each compression level (raw / JPEG-95 / JPEG-75 / JPEG-50) and each of
FFT / SigLIP2 / DINOv2:
  - "raw" reuses the EXISTING {prefix}{detector}_features.csv (these were
    extracted directly from uncompressed images) restricted to the
    canonical eval-set paths -- no re-extraction needed.
  - JPEG-95/75/50 are extracted FRESH from the on-disk images: load image
    (routing.extractors.common.sample_frame_pil) -> apply_jpeg_compression
    (same helper used by the original e7_compression.py) -> run EACH
    detector's exact extraction function (extract_fft.extract_features /
    SigLIP2 get_image_features / DINOv2 CLS token) on that SAME compressed
    image (one decode+compress per eval image, shared across all three
    detectors, for efficiency).

In every case, inference uses the EXISTING trained probe
({prefix}fft_probe.pkl / {prefix}siglip2_probe.pkl / {prefix}dinov2_probe.pkl)
loaded via joblib -- (clf, scaler) -- with NO retraining, exactly mirroring
how e7_compression.py evaluates CNNSpot/CLIP under compression using their
already-trained classifiers.

The original CNNSpot/CLIP rows from results/e7_compression.csv are copied
forward for continuity (note: those were measured on the OLD, smaller val
split -- they are NOT re-run here).

Output: routing/results/e7_compression_extended.csv (long format)
  columns: dataset, compression, jpeg_quality, detector, n_samples, accuracy, note

Usage:
    python e7_compression_extended.py --extract   # fresh-extract JPEG95/75/50 features+preds, cache to disk
    python e7_compression_extended.py --evaluate   # build the final CSV from cached + raw results
    (no flags = both)
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extractors"))
from extractors.common import (
    DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir,
    load_index, sample_frame_pil, apply_jpeg_compression, clear_gpu,
)
from extractors.extract_fft import extract_features as fft_extract_features
from pipeline.splits import get_canonical_eval_paths

CACHE_DIR = os.path.join(RESULTS_DIR, "compression_extended_cache")
COMPRESSION_LEVELS = {"raw": None, "jpeg95": 95, "jpeg75": 75, "jpeg50": 50}
ORIGINAL_E7_CSV = os.path.join(RESULTS_DIR, "e7_compression.csv")

DEVICE = "cuda"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
DINOV2_MODEL_ID = "facebook/dinov2-base"


def load_probe(ds_name, detector):
    cfg = DATASETS[ds_name]
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{detector}_probe.pkl")
    if not os.path.exists(probe_pkl):
        raise FileNotFoundError(f"Missing probe: {probe_pkl}")
    return joblib.load(probe_pkl)


def raw_accuracy(ds_name, detector, eval_paths):
    """Reuse existing {prefix}{detector}_features.csv for the 'raw' condition."""
    cfg = DATASETS[ds_name]
    feat_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{detector}_features.csv")
    df = pd.read_csv(feat_csv)
    df = df[df["path"].isin(eval_paths)].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)
    clf, scaler = load_probe(ds_name, detector)
    y_pred = clf.predict(scaler.transform(X))
    acc = float(np.mean(y_pred == y))
    return acc, len(df)


def extract_compressed_features_all3(ds_name, quality):
    """
    For one (dataset, quality), decode+compress each canonical-eval image
    ONCE and extract FFT + SigLIP2 + DINOv2 features from that same
    compressed image. Caches a per-detector feature CSV to CACHE_DIR.
    Returns dict[detector] -> (X, y, paths) in-memory as well.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg = DATASETS[ds_name]
    eval_paths = get_canonical_eval_paths(ds_name)
    index_rows = load_index(cfg["index"])
    eval_rows = [r for r in index_rows if r["path"] in eval_paths]

    cache_files = {
        det: os.path.join(CACHE_DIR, f"{cfg['prefix']}{det}_q{quality}_features.npz")
        for det in ["fft", "siglip2", "dinov2"]
    }
    if all(os.path.exists(f) for f in cache_files.values()):
        print(f"  [cache hit] {ds_name} q{quality}")
        out = {}
        for det, f in cache_files.items():
            d = np.load(f, allow_pickle=True)
            out[det] = (d["X"], d["y"], d["paths"])
        return out

    print(f"Loading SigLIP2 ({SIGLIP2_MODEL_ID}) and DINOv2 ({DINOV2_MODEL_ID})...")
    from transformers import AutoProcessor, AutoModel, AutoImageProcessor
    sig_proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
    sig_model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()
    din_proc = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    din_model = AutoModel.from_pretrained(DINOV2_MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()

    feats_fft, feats_sig, feats_din, labels, paths = [], [], [], [], []
    for r in tqdm(eval_rows, desc=f"{ds_name} extract @ JPEG-{quality}"):
        path = r["path"]
        label = int(r["label"])
        try:
            img = sample_frame_pil(path, cfg["is_video"])
            img_c = apply_jpeg_compression(img, quality)
        except Exception as e:
            print(f"  Skip {path}: {e}")
            continue

        feats_fft.append(fft_extract_features(img_c))

        inputs = sig_proc(images=img_c, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            f = sig_model.get_image_features(**inputs)
        f = f[0].float().flatten()
        if f.shape[0] != 768:
            f = f.reshape(-1, 768).mean(dim=0)
        feats_sig.append(f.cpu().numpy())

        inputs2 = din_proc(images=img_c, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = din_model(**inputs2)
        feats_din.append(out.last_hidden_state[0, 0].float().cpu().numpy())

        labels.append(label)
        paths.append(path)

    del sig_model, din_model
    clear_gpu()

    y_arr = np.array(labels)
    paths_arr = np.array(paths, dtype=object)
    out = {
        "fft": (np.stack(feats_fft).astype(np.float32), y_arr, paths_arr),
        "siglip2": (np.stack(feats_sig).astype(np.float32), y_arr, paths_arr),
        "dinov2": (np.stack(feats_din).astype(np.float32), y_arr, paths_arr),
    }
    for det, (X, y, p) in out.items():
        np.savez(cache_files[det], X=X, y=y, paths=p)
    print(f"  Cached features for {ds_name} q{quality} -> {CACHE_DIR}")
    return out


def compressed_accuracy(ds_name, detector, quality, cached):
    X, y, _ = cached[detector]
    clf, scaler = load_probe(ds_name, detector)
    y_pred = clf.predict(scaler.transform(X))
    return float(np.mean(y_pred == y)), len(y)


def copy_forward_original_rows():
    """Reshape the original e7_compression.csv (CNNSpot/CLIP, old val split)
    into the new long-format schema for continuity."""
    if not os.path.exists(ORIGINAL_E7_CSV):
        print(f"  Original {ORIGINAL_E7_CSV} not found -- skipping copy-forward.")
        return []
    df = pd.read_csv(ORIGINAL_E7_CSV)
    rows = []
    note = ("copied from original e7_compression.csv -- OLD pre-expansion "
            "val split (n~400/480/480), NOT re-run on current canonical splits")
    for _, r in df.iterrows():
        rows.append({
            "dataset": r["dataset"], "compression": r["compression"],
            "jpeg_quality": r["jpeg_quality"], "detector": "CNNSpot",
            "n_samples": r["n_samples"], "accuracy": r["cnnspot_accuracy"],
            "note": note,
        })
        rows.append({
            "dataset": r["dataset"], "compression": r["compression"],
            "jpeg_quality": r["jpeg_quality"], "detector": "CLIP",
            "n_samples": r["n_samples"], "accuracy": r["clip_accuracy"],
            "note": note,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()
    do_extract = args.extract or not (args.extract or args.evaluate)
    do_evaluate = args.evaluate or not (args.extract or args.evaluate)

    ensure_results_dir()
    rows = copy_forward_original_rows()

    for ds_name in args.datasets:
        eval_paths = get_canonical_eval_paths(ds_name)
        print(f"\n=== {ds_name} (n_eval={len(eval_paths)}) ===")

        # raw
        for detector in ["fft", "siglip2", "dinov2"]:
            acc, n = raw_accuracy(ds_name, detector, eval_paths)
            rows.append({
                "dataset": ds_name, "compression": "raw", "jpeg_quality": "raw",
                "detector": detector.upper() if detector != "fft" else "FFT",
                "n_samples": n, "accuracy": acc,
                "note": "current canonical eval split; reused existing raw features",
            })
            print(f"  raw {detector}: acc={acc:.4f} (n={n})")

        for level, quality in COMPRESSION_LEVELS.items():
            if quality is None:
                continue
            if do_extract:
                cached = extract_compressed_features_all3(ds_name, quality)
            else:
                cached = extract_compressed_features_all3(ds_name, quality)  # cache-hit path
            if not do_evaluate:
                continue
            for detector in ["fft", "siglip2", "dinov2"]:
                acc, n = compressed_accuracy(ds_name, detector, quality, cached)
                rows.append({
                    "dataset": ds_name, "compression": level, "jpeg_quality": quality,
                    "detector": detector.upper() if detector != "fft" else "FFT",
                    "n_samples": n, "accuracy": acc,
                    "note": "current canonical eval split; freshly re-extracted under JPEG compression",
                })
                print(f"  {level} {detector}: acc={acc:.4f} (n={n})")

    if do_evaluate:
        out = pd.DataFrame(rows)
        out_csv = os.path.join(RESULTS_DIR, "e7_compression_extended.csv")
        out.to_csv(out_csv, index=False)
        print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
