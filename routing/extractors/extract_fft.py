"""
Dual-stream FFT artifact detector (A3).
Computes spatial + frequency domain statistics as features,
then trains a logistic regression probe.

Stream 1: Spatial — DCT coefficient statistics (blocking artifacts, sharpness).
Stream 2: Frequency — Radial FFT magnitude profiles (periodic GAN/diffusion artifacts).

Usage:
    python extract_fft.py --dataset GenBuster
    python extract_fft.py --dataset SD14
    python extract_fft.py --dataset BigGAN
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import load_index, sample_frame_pil, DATASETS, ROUTING_BASE


N_RADIAL_BINS = 32
DCT_PATCH = 8
N_SPATIAL_STATS = 8  # mean, std, skew, kurt, p5, p25, p75, p95 of high-freq DCT


def radial_fft_profile(gray_f32, n_bins=N_RADIAL_BINS):
    """Compute normalized radial magnitude spectrum as a 1-D feature vector."""
    f = np.fft.fft2(gray_f32)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2).astype(np.float32)
    r_max = np.sqrt(cy ** 2 + cx ** 2)
    r_norm = r / (r_max + 1e-8)

    bins = np.linspace(0, 1, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
        if mask.sum() > 0:
            profile[i] = mag[mask].mean()
    # L2-normalize
    norm = np.linalg.norm(profile) + 1e-8
    return profile / norm


def dct_stats(gray_u8):
    """Compute statistics of high-frequency DCT coefficients on 8x8 blocks."""
    h, w = gray_u8.shape
    # Pad to multiples of DCT_PATCH
    ph = (h // DCT_PATCH) * DCT_PATCH
    pw = (w // DCT_PATCH) * DCT_PATCH
    img = gray_u8[:ph, :pw].astype(np.float32) - 128.0

    hf_coeffs = []
    for i in range(0, ph, DCT_PATCH):
        for j in range(0, pw, DCT_PATCH):
            block = img[i:i + DCT_PATCH, j:j + DCT_PATCH]
            dct = cv2.dct(block)
            # High-frequency: bottom-right 3x3 of the 8x8 DCT block
            hf_coeffs.extend(dct[5:, 5:].flatten().tolist())

    hf = np.array(hf_coeffs, dtype=np.float32)
    from scipy.stats import skew, kurtosis
    stats = np.array([
        hf.mean(), hf.std(),
        float(skew(hf)), float(kurtosis(hf)),
        np.percentile(hf, 5), np.percentile(hf, 25),
        np.percentile(hf, 75), np.percentile(hf, 95),
    ], dtype=np.float32)
    return stats


def color_coherence(img_rgb):
    """Channel correlation statistics (color forgery cue)."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    feats = np.array([
        np.corrcoef(r.flatten(), g.flatten())[0, 1],
        np.corrcoef(r.flatten(), b.flatten())[0, 1],
        np.corrcoef(g.flatten(), b.flatten())[0, 1],
    ], dtype=np.float32)
    return feats


def extract_features(img_pil, target_size=256):
    """Full dual-stream feature extraction pipeline."""
    img_pil = img_pil.resize((target_size, target_size))
    img_rgb = np.array(img_pil, dtype=np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_f32 = gray.astype(np.float32) / 255.0

    fft_feats = radial_fft_profile(gray_f32)          # N_RADIAL_BINS
    dct_feats = dct_stats(gray)                        # N_SPATIAL_STATS
    color_feats = color_coherence(img_rgb)             # 3

    return np.concatenate([fft_feats, dct_feats, color_feats])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]

    out_file = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_features.csv")
    if os.path.exists(out_file):
        print(f"Already exists: {out_file}")
        return

    index_rows = load_index(cfg["index"])
    print(f"Extracting FFT features for {len(index_rows)} samples ({args.dataset})...")

    # Test feature dim on a dummy image
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    from PIL import Image as PILImage
    dummy_feats = extract_features(PILImage.fromarray(dummy))
    feat_dim = len(dummy_feats)
    print(f"Feature dimension: {feat_dim}")

    rows_out = []
    for r in tqdm(index_rows):
        path = r["path"]
        label = int(r["label"])
        generator = r.get("generator", "unknown")

        try:
            img = sample_frame_pil(path, cfg["is_video"])
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        t0 = time.perf_counter()
        feats = extract_features(img)
        latency_ms = (time.perf_counter() - t0) * 1000

        rows_out.append({
            "path": path,
            "y_true": label,
            "generator": generator,
            **{f"f{i}": feats[i] for i in range(len(feats))},
            "latency_fft_ms": latency_ms,
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    main()
