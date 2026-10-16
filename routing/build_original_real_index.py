"""
Build PARALLEL index CSVs pairing the EXISTING SDXL/FLUX fake images
(from routing/sdxl_index.csv / routing/flux_index.csv) with a sample of the
ORIGINAL AIGIBench-bundled PASCAL VOC real images instead of the
ImageNet/SD1.4 real pool currently used.

This exists to answer the reviewers' request (peerreviewdecision2.txt,
Reviewer 1 point 4) to report results on the ORIGINAL real-image benchmark
alongside the modified (ImageNet) one.

Mirrors the sampling logic in build_aigibench_index.py (same seed=42, same
n_samples=2000 per class, same find_images() helper with only_subdir
filtering) but:
  - REAL_DIR points at the original data/aigibench/real/<VOC class>/0_real
    pool (4000 images total: 20 classes x 200) instead of ImageNet val/nature.
  - Fake images are NOT re-sampled: they are read directly from the existing
    sdxl_index.csv / flux_index.csv so the fake side of the comparison is
    IDENTICAL to the already-computed SDXL/FLUX feature/prediction files,
    letting us reuse those without recomputing.

Does NOT overwrite sdxl_index.csv / flux_index.csv. Writes:
  routing/results/sdxl_index_original_real.csv
  routing/results/flux_index_original_real.csv
"""
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from build_aigibench_index import find_images  # reuse, don't reimplement

random.seed(42)

ROUTING    = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROUTING, "results")
ORIG_REAL_DIR = "/workspace/benchmarks_ai_research/data/aigibench/real"
N_SAMPLES  = 2000  # matches current sdxl_index.csv / flux_index.csv per-class size

EXISTING = [
    ("sdxl_index.csv", "sdxl_index_original_real.csv"),
    ("flux_index.csv", "flux_index_original_real.csv"),
]


def load_fake_rows(index_csv):
    with open(os.path.join(ROUTING, index_csv)) as f:
        rows = list(csv.DictReader(f))
    fake = [r for r in rows if int(r["label"]) == 1]
    return fake


def main():
    os.makedirs(RESULTS, exist_ok=True)

    print(f"Scanning ORIGINAL PASCAL VOC real images (0_real/ subdirs) under {ORIG_REAL_DIR} ...")
    real_imgs = find_images(ORIG_REAL_DIR, only_subdir="0_real")
    print(f"  Found {len(real_imgs)} original real images")
    if len(real_imgs) == 0:
        print("ERROR: no original real images found — check extraction.")
        return

    for existing_csv, out_name in EXISTING:
        fake_rows = load_fake_rows(existing_csv)
        n = min(N_SAMPLES, len(real_imgs))
        real_sample = random.sample(real_imgs, n)  # continues global random stream,
        # mirroring build_index() being called once per generator in sequence
        # in the original script.

        rows = (
            [{"path": p, "label": 0, "generator": "real_voc_original"} for p in real_sample] +
            [{"path": r["path"], "label": 1, "generator": r["generator"]} for r in fake_rows]
        )
        random.shuffle(rows)

        out_path = os.path.join(RESULTS, out_name)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
            writer.writeheader()
            writer.writerows(rows)

        n_real = sum(1 for r in rows if r["label"] == 0)
        n_fake = sum(1 for r in rows if r["label"] == 1)
        print(f"  Saved {out_path} (real={n_real} [original VOC], fake={n_fake} [existing {existing_csv} fakes], total={len(rows)})")

        # Also save a REAL-ONLY index (no fake rows) so feature extraction
        # only needs to run on the new real images -- fake features already
        # exist in the current sdxl_*/flux_* feature CSVs and are reused as-is.
        real_only_name = out_name.replace("_index_original_real.csv", "_realonly_original.csv")
        real_only_path = os.path.join(RESULTS, real_only_name)
        with open(real_only_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
            writer.writeheader()
            writer.writerows([{"path": p, "label": 0, "generator": "real_voc_original"} for p in real_sample])
        print(f"  Saved {real_only_path} (real-only index, n={len(real_sample)})")


if __name__ == "__main__":
    main()
