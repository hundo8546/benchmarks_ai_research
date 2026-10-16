"""
routing/pipeline/robustness/e9_postprocessing_robustness.py

Peer reviewer 1 (claude_context_notes/peerreviewdecision2.txt, point 9) asked
to test robustness to "common post-processing operations" before trusting the
FFT artifact detector's near-perfect accuracy. JPEG compression was already
covered (pipeline/robustness/e7_compression_extended.py). This script covers
the two other most common, realistic post-processing operations an image
might undergo between generation and being scored by a detector:

  1. Gaussian blur (simulates re-upload / screenshot / re-compression blur),
     at sigma in {0.0 [none/baseline], 1.0, 2.0}, applied to the raw PIL
     image via PIL.ImageFilter.GaussianBlur BEFORE the standard
     256x256-resize-then-extract pipeline each detector already uses.

  2. Downscale-then-upscale (simulates a social-media/messaging-app resize
     pipeline: image downsized for transmission, then upscaled back by the
     viewer/platform), at downscale factor in {1.0 [none], 0.5, 0.25}, using
     bicubic interpolation both ways, applied to the raw PIL image BEFORE the
     standard pipeline.

Exactly mirrors e7_compression_extended.py's structure:
  - "none" severity (sigma=0.0 / factor=1.0) reuses the EXISTING
    {prefix}{detector}_features.csv (extracted from un-perturbed images),
    restricted to the canonical eval-set paths (routing.pipeline.splits) --
    no re-extraction needed. This should numerically match e7's "raw" rows
    (same probe, same eval split, same un-perturbed features) and serves as
    a built-in sanity check.
  - Non-"none" severities are extracted FRESH from on-disk images: load
    image (extractors.common.sample_frame_pil) -> apply perturbation ->
    run EACH detector's exact extraction function (extract_fft.extract_features
    / SigLIP2 get_image_features / DINOv2 CLS token) on that SAME perturbed
    image (one decode+perturb per eval image, shared across all three
    detectors, for efficiency).
  - In every case, inference uses the EXISTING trained probe
    ({prefix}fft_probe.pkl / {prefix}siglip2_probe.pkl /
    {prefix}dinov2_probe.pkl) loaded via joblib -- (clf, scaler) -- with NO
    retraining.

Output: routing/results/e9_postprocessing_robustness.csv (long format)
  columns: dataset, detector, perturbation_type, severity_level, n_samples,
  accuracy
  (deliberately the same column *shape* as e7_compression_extended.csv, with
  compression/jpeg_quality replaced by perturbation_type/severity_level, so
  the two CSVs can be concatenated/compared directly.)

Usage:
    python e9_postprocessing_robustness.py --extract   # fresh-extract perturbed features+preds, cache to disk
    python e9_postprocessing_robustness.py --evaluate  # build the final CSV from cached + raw results
    (no flags = both)
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extractors"))
from extractors.common import (
    DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir,
    load_index, sample_frame_pil, clear_gpu,
)
from extractors.extract_fft import extract_features as fft_extract_features
from pipeline.splits import get_canonical_eval_paths

CACHE_DIR = os.path.join(RESULTS_DIR, "postprocessing_robustness_cache")

DEVICE = "cuda"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
DINOV2_MODEL_ID = "facebook/dinov2-base"


# -----------------------------------------------------------------------
# Perturbations
# -----------------------------------------------------------------------

def apply_gaussian_blur(img, sigma):
    """Apply Gaussian blur to a PIL Image and return the blurred PIL Image."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_downscale_upscale(img, factor):
    """Downscale a PIL Image by `factor` then upscale back to the original
    dimensions (bicubic both ways) -- simulates a transmit-then-view resize
    pipeline. factor=1.0 is a no-op."""
    if factor >= 1.0:
        return img
    w, h = img.size
    small_w = max(1, round(w * factor))
    small_h = max(1, round(h * factor))
    small = img.resize((small_w, small_h), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


# severity_level -> (perturb_fn_param, is_baseline)
GAUSSIAN_BLUR_LEVELS = {
    "sigma_0.0": 0.0,
    "sigma_1.0": 1.0,
    "sigma_2.0": 2.0,
}
DOWNSCALE_UPSCALE_LEVELS = {
    "factor_1.0": 1.0,
    "factor_0.5": 0.5,
    "factor_0.25": 0.25,
}

PERTURBATIONS = {
    "gaussian_blur": {
        "levels": GAUSSIAN_BLUR_LEVELS,
        "baseline_key": "sigma_0.0",
        "apply": apply_gaussian_blur,
    },
    "downscale_upscale": {
        "levels": DOWNSCALE_UPSCALE_LEVELS,
        "baseline_key": "factor_1.0",
        "apply": apply_downscale_upscale,
    },
}


# -----------------------------------------------------------------------
# Probe / accuracy helpers (mirrors e7_compression_extended.py)
# -----------------------------------------------------------------------

def load_probe(ds_name, detector):
    cfg = DATASETS[ds_name]
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{detector}_probe.pkl")
    if not os.path.exists(probe_pkl):
        raise FileNotFoundError(f"Missing probe: {probe_pkl}")
    return joblib.load(probe_pkl)


def raw_accuracy(ds_name, detector, eval_paths):
    """Reuse existing {prefix}{detector}_features.csv for the baseline
    (un-perturbed) condition -- identical to e7_compression_extended.py's
    raw_accuracy()."""
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


def extract_perturbed_features_all3(ds_name, perturbation_type, severity_key):
    """
    For one (dataset, perturbation_type, severity_key), decode+perturb each
    canonical-eval image ONCE and extract FFT + SigLIP2 + DINOv2 features
    from that same perturbed image. Caches a per-detector feature .npz to
    CACHE_DIR. Returns dict[detector] -> (X, y, paths).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg = DATASETS[ds_name]
    eval_paths = get_canonical_eval_paths(ds_name)
    index_rows = load_index(cfg["index"])
    eval_rows = [r for r in index_rows if r["path"] in eval_paths]

    spec = PERTURBATIONS[perturbation_type]
    param = spec["levels"][severity_key]
    apply_fn = spec["apply"]

    cache_files = {
        det: os.path.join(
            CACHE_DIR,
            f"{cfg['prefix']}{det}_{perturbation_type}_{severity_key}_features.npz",
        )
        for det in ["fft", "siglip2", "dinov2"]
    }
    if all(os.path.exists(f) for f in cache_files.values()):
        print(f"  [cache hit] {ds_name} {perturbation_type}/{severity_key}")
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
    for r in tqdm(eval_rows, desc=f"{ds_name} extract @ {perturbation_type}/{severity_key}"):
        path = r["path"]
        label = int(r["label"])
        try:
            img = sample_frame_pil(path, cfg["is_video"])
            img_p = apply_fn(img, param)
        except Exception as e:
            print(f"  Skip {path}: {e}")
            continue

        feats_fft.append(fft_extract_features(img_p))

        inputs = sig_proc(images=img_p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            f = sig_model.get_image_features(**inputs)
        f = f[0].float().flatten()
        if f.shape[0] != 768:
            f = f.reshape(-1, 768).mean(dim=0)
        feats_sig.append(f.cpu().numpy())

        inputs2 = din_proc(images=img_p, return_tensors="pt").to(DEVICE)
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
    print(f"  Cached features for {ds_name} {perturbation_type}/{severity_key} -> {CACHE_DIR}")
    return out


def perturbed_accuracy(ds_name, detector, cached):
    X, y, paths = cached[detector]
    finite_mask = np.isfinite(X).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        # Aggressive perturbations (e.g. downscale-then-upscale at factor=0.25 on an
        # already-small native image) can occasionally collapse a color channel to
        # near-constant, making the FFT color-coherence feature (a Pearson
        # correlation) divide by ~zero variance and return NaN. Drop those rows
        # rather than crash or silently impute a value -- report the exact count so
        # it's auditable, not hidden.
        bad_paths = paths[~finite_mask][:5]
        print(f"    WARNING: {ds_name}/{detector}: dropping {n_dropped}/{len(y)} samples "
              f"with non-finite features (e.g. {list(bad_paths)})")
    X, y = X[finite_mask], y[finite_mask]
    clf, scaler = load_probe(ds_name, detector)
    y_pred = clf.predict(scaler.transform(X))
    return float(np.mean(y_pred == y)), len(y), n_dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()
    do_extract = args.extract or not (args.extract or args.evaluate)
    do_evaluate = args.evaluate or not (args.extract or args.evaluate)

    ensure_results_dir()
    rows = []

    for ds_name in args.datasets:
        eval_paths = get_canonical_eval_paths(ds_name)
        print(f"\n=== {ds_name} (n_eval={len(eval_paths)}) ===")

        for perturbation_type, spec in PERTURBATIONS.items():
            for severity_key in spec["levels"]:
                if severity_key == spec["baseline_key"]:
                    # Baseline ("none") -- reuse existing raw features, no
                    # re-extraction needed (identical to e7's "raw" rows).
                    for detector in ["fft", "siglip2", "dinov2"]:
                        acc, n = raw_accuracy(ds_name, detector, eval_paths)
                        rows.append({
                            "dataset": ds_name,
                            "detector": detector.upper() if detector != "fft" else "FFT",
                            "perturbation_type": perturbation_type,
                            "severity_level": severity_key,
                            "n_samples": n,
                            "n_dropped_nonfinite": 0,
                            "accuracy": acc,
                        })
                        print(f"  [baseline] {perturbation_type}/{severity_key} {detector}: acc={acc:.4f} (n={n})")
                    continue

                if do_extract:
                    cached = extract_perturbed_features_all3(ds_name, perturbation_type, severity_key)
                else:
                    cached = extract_perturbed_features_all3(ds_name, perturbation_type, severity_key)  # cache-hit path
                if not do_evaluate:
                    continue
                for detector in ["fft", "siglip2", "dinov2"]:
                    acc, n, n_dropped = perturbed_accuracy(ds_name, detector, cached)
                    rows.append({
                        "dataset": ds_name,
                        "detector": detector.upper() if detector != "fft" else "FFT",
                        "perturbation_type": perturbation_type,
                        "severity_level": severity_key,
                        "n_samples": n,
                        "n_dropped_nonfinite": n_dropped,
                        "accuracy": acc,
                    })
                    print(f"  {perturbation_type}/{severity_key} {detector}: acc={acc:.4f} (n={n}, dropped={n_dropped})")

        # Save after EVERY dataset, not just at the very end -- a crash on dataset
        # N no longer erases the completed results for datasets 1..N-1 (this is
        # exactly what happened on the first run: it reached 11/12 passes before
        # crashing on a NaN in the final BigGAN severity, and lost everything
        # because nothing had been written to disk yet).
        if do_evaluate and rows:
            out = pd.DataFrame(rows, columns=[
                "dataset", "detector", "perturbation_type", "severity_level",
                "n_samples", "n_dropped_nonfinite", "accuracy",
            ])
            out_csv = os.path.join(RESULTS_DIR, "e9_postprocessing_robustness.csv")
            out.to_csv(out_csv, index=False)
            print(f"  [checkpoint] Saved {len(rows)} rows so far -> {out_csv}")

    if do_evaluate:
        out = pd.DataFrame(rows, columns=[
            "dataset", "detector", "perturbation_type", "severity_level",
            "n_samples", "n_dropped_nonfinite", "accuracy",
        ])
        out_csv = os.path.join(RESULTS_DIR, "e9_postprocessing_robustness.csv")
        out.to_csv(out_csv, index=False)
        print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
