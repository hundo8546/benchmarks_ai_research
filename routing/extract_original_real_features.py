"""
Extract FFT, SigLIP2, DINOv2 features for the ORIGINAL PASCAL VOC real
images sampled by build_original_real_index.py (routing/results/
sdxl_realonly_original.csv, flux_realonly_original.csv).

We only need features for the new real images -- the fake-image features
for SDXL/FLUX already exist in routing/sdxl_{fft,siglip2,dinov2}_features.csv
and routing/flux_{fft,siglip2,dinov2}_features.csv and are reused unchanged.

Mirrors the exact extraction code in routing/experiments/e8_modern_diffusion.py
(extract_features()) for SigLIP2/DINOv2/FFT -- same model IDs, same pooling,
same float32 (not fp16) -- so the new real-image features are directly
comparable/mergeable with the existing fake-image feature CSVs produced by
that same code path. CLIP is skipped (not needed for this analysis).

Writes (does NOT touch existing routing/sdxl_*_features.csv / routing/flux_*_features.csv):
  routing/results/sdxl_realonly_fft_features.csv
  routing/results/sdxl_realonly_siglip2_features.csv
  routing/results/sdxl_realonly_dinov2_features.csv
  routing/results/flux_realonly_fft_features.csv
  routing/results/flux_realonly_siglip2_features.csv
  routing/results/flux_realonly_dinov2_features.csv
"""
import csv
import os
import sys

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROUTING_BASE = "/workspace/benchmarks_ai_research/routing"
RESULTS = os.path.join(ROUTING_BASE, "results")
sys.path.insert(0, ROUTING_BASE)
sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
from extractors.extract_fft import extract_features as fft_compute  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDICES = [
    ("sdxl", os.path.join(RESULTS, "sdxl_realonly_original.csv")),
    ("flux", os.path.join(RESULTS, "flux_realonly_original.csv")),
]

if "--coco" in sys.argv:
    INDICES = [
        ("sdxl_coco", os.path.join(RESULTS, "sdxl_realonly_coco.csv")),
        ("flux_coco", os.path.join(RESULTS, "flux_realonly_coco.csv")),
    ]


def load_index(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_fft_for(rows, out_path):
    if os.path.exists(out_path):
        print(f"  FFT already exists: {out_path}")
        return
    out_rows = []
    for r in tqdm(rows, desc="FFT"):
        try:
            img = Image.open(r["path"]).convert("RGB")
            feats = fft_compute(img)
            row = {"path": r["path"], "y_true": int(r["label"]), "generator": r["generator"]}
            row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
            out_rows.append(row)
        except Exception as e:
            print(f"    skip {r['path']}: {e}")
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(out_rows)} rows)")


def extract_siglip2_for(rows, out_path):
    if os.path.exists(out_path):
        print(f"  SigLIP2 already exists: {out_path}")
        return
    from transformers import AutoProcessor, AutoModel
    proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(DEVICE).eval()
    out_rows = []
    for r in tqdm(rows, desc="SigLIP2"):
        try:
            img = Image.open(r["path"]).convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                f = model.get_image_features(**inp)[0].float()
                if f.numel() != 768:
                    f = f.flatten().reshape(-1, 768).mean(dim=0)
                feats = f.cpu().numpy().flatten()
            row = {"path": r["path"], "y_true": int(r["label"]), "generator": r["generator"]}
            row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
            out_rows.append(row)
        except Exception as e:
            print(f"    skip {r['path']}: {e}")
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(out_rows)} rows)")


def extract_dinov2_for(rows, out_path):
    if os.path.exists(out_path):
        print(f"  DINOv2 already exists: {out_path}")
        return
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    out_rows = []
    for r in tqdm(rows, desc="DINOv2"):
        try:
            img = Image.open(r["path"]).convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
                feats = out.last_hidden_state[:, 0].float().cpu().numpy().flatten()
            row = {"path": r["path"], "y_true": int(r["label"]), "generator": r["generator"]}
            row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
            out_rows.append(row)
        except Exception as e:
            print(f"    skip {r['path']}: {e}")
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(out_rows)} rows)")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for tag, index_path in INDICES:
        print(f"\n=== {tag} real-only features ===")
        rows = load_index(index_path)
        print(f"  {len(rows)} real images")
        extract_fft_for(rows, os.path.join(RESULTS, f"{tag}_realonly_fft_features.csv"))
        extract_siglip2_for(rows, os.path.join(RESULTS, f"{tag}_realonly_siglip2_features.csv"))
        extract_dinov2_for(rows, os.path.join(RESULTS, f"{tag}_realonly_dinov2_features.csv"))
    print("\nDone.")


if __name__ == "__main__":
    main()
