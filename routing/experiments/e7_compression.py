"""
E7 — Compression Robustness Study.
Measures detector degradation under JPEG compression at:
  Raw, JPEG-95, JPEG-75, JPEG-50

Evaluates all detector pairings on compressed versions of the val set.
This addresses reviewer concern about real-world deployment degradation.

Produces: results/e7_compression.csv

Usage:
    # Extract features under compression first, then evaluate
    python e7_compression.py --extract --dataset GenBuster
    python e7_compression.py --evaluate
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import (
    DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir,
    load_index, sample_frame_pil, apply_jpeg_compression, clear_gpu, load_val_paths,
)

COMPRESSION_LEVELS = {
    "raw": None,        # No compression
    "jpeg95": 95,
    "jpeg75": 75,
    "jpeg50": 50,
}

LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4


def extract_compressed_clip_preds(ds_name, quality, out_csv):
    """Extract CLIP predictions on compressed images."""
    from transformers import CLIPProcessor, CLIPModel
    import joblib

    cfg = DATASETS[ds_name]

    # Load existing CLIP features & classifier
    feat_file = os.path.join(ROUTING_BASE, f"{cfg['prefix']}clip_features.csv")
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}clip_classifier.pkl")
    if not os.path.exists(probe_pkl):
        probe_pkl = os.path.join(ROUTING_BASE, "clip_classifier.pkl")

    if not os.path.exists(feat_file) or not os.path.exists(probe_pkl):
        print(f"  Missing CLIP features or probe for {ds_name}, skipping")
        return

    clf, scaler = joblib.load(probe_pkl)

    DEVICE = "cuda"
    MODEL_ID = "openai/clip-vit-base-patch32"
    proc = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    index_rows = load_index(cfg["index"])
    all_paths = [r["path"] for r in index_rows]
    vp = load_val_paths(cfg, all_paths)
    val_rows = [r for r in index_rows if r["path"] in vp]

    rows_out = []
    for r in tqdm(val_rows, desc=f"CLIP@Q{quality}"):
        path = r["path"]
        try:
            img = sample_frame_pil(path, cfg["is_video"])
            if quality is not None:
                img = apply_jpeg_compression(img, quality)
        except Exception as e:
            continue

        inputs = proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)[0].float().cpu().numpy().flatten()

        feat_scaled = scaler.transform(feats.reshape(1, -1))
        y_clip = int(clf.predict(feat_scaled)[0])
        clip_proba = float(clf.predict_proba(feat_scaled)[0, 1])

        rows_out.append({
            "path": path,
            "y_true": int(r["label"]),
            "y_clip": y_clip,
            "clip_proba": clip_proba,
        })

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")


def extract_compressed_cnnspot_preds(ds_name, quality, out_csv):
    """Run CNNSpot on compressed images."""
    import sys
    sys.path.insert(0, "/workspace/benchmarks_ai_research/CNNDetection")
    from networks.resnet import resnet50

    DEVICE = "cuda"
    MODEL_PATH = "/workspace/benchmarks_ai_research/weights/blur_jpg_prob0.1.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"  CNNSpot weights not found: {MODEL_PATH}")
        return

    model = resnet50(num_classes=1)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(DEVICE).eval()

    cfg = DATASETS[ds_name]
    index_rows = load_index(cfg["index"])
    all_paths = [r["path"] for r in index_rows]
    vp = load_val_paths(cfg, all_paths)
    val_rows = [r for r in index_rows if r["path"] in vp]

    import cv2
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    rows_out = []
    for r in tqdm(val_rows, desc=f"CNN@Q{quality}"):
        path = r["path"]
        try:
            img = sample_frame_pil(path, cfg["is_video"])
            if quality is not None:
                img = apply_jpeg_compression(img, quality)
        except Exception as e:
            continue

        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor).item()
        prob_fake = torch.sigmoid(torch.tensor(logit)).item()
        y_pred = 1 if prob_fake > 0.5 else 0
        rows_out.append({
            "path": path,
            "y_true": int(r["label"]),
            "y_cnn": y_pred,
            "prob_fake": prob_fake,
        })

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")


def evaluate_compression(datasets):
    """Evaluate all detector pairings across compression levels."""
    ensure_results_dir()
    all_rows = []

    for ds_name in datasets:
        cfg = DATASETS[ds_name]
        print(f"\n=== {ds_name} ===")

        for level, quality in COMPRESSION_LEVELS.items():
            q_tag = str(quality) if quality is not None else "raw"

            # Load CNN preds (compressed or original)
            if quality is None:
                cnn_csv = cfg["cnnspot_csv"]
            else:
                cnn_csv = os.path.join(ROUTING_BASE,
                                       f"{cfg['prefix']}cnnspot_q{quality}.csv")

            if quality is None:
                clip_csv = cfg["clip_preds"]
            else:
                clip_csv = os.path.join(ROUTING_BASE,
                                        f"{cfg['prefix']}clip_q{quality}.csv")

            if not os.path.exists(cnn_csv) or not os.path.exists(clip_csv):
                print(f"  {level}: missing CSV, skipping")
                continue

            df_cnn = pd.read_csv(cnn_csv)
            df_clip = pd.read_csv(clip_csv)

            df = df_cnn.merge(df_clip[["path", "y_clip"]], on="path", how="inner")
            val_paths = load_val_paths(cfg, df["path"].tolist())
            df = df[df["path"].isin(val_paths)].reset_index(drop=True)
            if len(df) == 0:
                continue

            y_true = df["y_true"].values
            y_cnn = df["y_cnn"].values
            y_clip = df["y_clip"].values
            disagree = (y_cnn != y_clip).astype(bool)

            cnn_acc = float(np.mean(y_cnn == y_true))
            clip_acc = float(np.mean(y_clip == y_true))
            agree_acc = float(np.mean(y_clip[~disagree] == y_true[~disagree])) if (~disagree).sum() > 0 else float("nan")
            dis_rate = float(np.mean(disagree))

            row = {
                "dataset": ds_name,
                "compression": level,
                "jpeg_quality": quality if quality is not None else "raw",
                "n_samples": len(df),
                "cnnspot_accuracy": cnn_acc,
                "clip_accuracy": clip_acc,
                "cheap_path_accuracy": agree_acc,
                "disagreement_rate": dis_rate,
            }
            all_rows.append(row)
            print(f"  {level:>8}: CNN={cnn_acc:.3f}, CLIP={clip_acc:.3f}, "
                  f"CheapPath={agree_acc:.3f}, DisRate={dis_rate:.3f}")

    df_out = pd.DataFrame(all_rows)
    out_file = os.path.join(RESULTS_DIR, "e7_compression.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")

    # Show degradation from raw to jpeg50
    print("\n=== Degradation: raw -> jpeg50 ===")
    for ds_name in datasets:
        sub = df_out[df_out["dataset"] == ds_name]
        raw = sub[sub["compression"] == "raw"]
        j50 = sub[sub["compression"] == "jpeg50"]
        if len(raw) == 0 or len(j50) == 0:
            continue
        cnn_drop = float(raw["cnnspot_accuracy"].iloc[0]) - float(j50["cnnspot_accuracy"].iloc[0])
        clip_drop = float(raw["clip_accuracy"].iloc[0]) - float(j50["clip_accuracy"].iloc[0])
        print(f"  {ds_name}: CNN drop={cnn_drop:+.3f}, CLIP drop={clip_drop:+.3f}")

    print("\nE7 complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true",
                   help="Extract features at each compression level")
    p.add_argument("--evaluate", action="store_true",
                   help="Evaluate all detectors across compression levels")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--detectors", nargs="+", default=["CNNSpot", "CLIP"])
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()

    if args.extract:
        for ds_name in args.datasets:
            cfg = DATASETS[ds_name]
            if not os.path.exists(cfg.get("val_paths", "")):
                print(f"Skipping {ds_name}: no val_paths")
                continue
            for level, quality in COMPRESSION_LEVELS.items():
                if quality is None:
                    continue  # raw = already extracted
                print(f"\nExtracting {ds_name} at JPEG-{quality}...")
                if "CNNSpot" in args.detectors:
                    out_cnn = os.path.join(ROUTING_BASE,
                                           f"{cfg['prefix']}cnnspot_q{quality}.csv")
                    if not os.path.exists(out_cnn):
                        extract_compressed_cnnspot_preds(ds_name, quality, out_cnn)
                if "CLIP" in args.detectors:
                    out_clip = os.path.join(ROUTING_BASE,
                                            f"{cfg['prefix']}clip_q{quality}.csv")
                    if not os.path.exists(out_clip):
                        extract_compressed_clip_preds(ds_name, quality, out_clip)

    if args.evaluate or not args.extract:
        evaluate_compression(args.datasets)


if __name__ == "__main__":
    main()
