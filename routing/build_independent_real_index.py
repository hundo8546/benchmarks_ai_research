"""
Build PARALLEL index CSVs pairing the EXISTING SDXL/FLUX fake images with a
sample of an INDEPENDENT real-image source (COCO val2017), distinct from
both PASCAL VOC (AIGIBench bundled) and the ImageNet/SD1.4 pool reused
across SD1.4/SDXL/FLUX.

Addresses reviewer request (peerreviewdecision2.txt, Reviewer 1 point 4):
"evaluate independently chosen real-image sources with varying compression,
resolution, and acquisition history."

Mirrors build_original_real_index.py's approach exactly (same seed=42, same
n_samples=2000 per generator, fake images taken AS-IS from the existing
sdxl_index.csv / flux_index.csv so existing fake-image features/predictions
are reused unchanged) but with REAL_DIR pointed at the downloaded COCO
val2017 sample instead of PASCAL VOC.

COCO val2017: 5000 images, Flickr-sourced, CC-licensed, varied resolution
and JPEG compression history, downloaded to
  /workspace/benchmarks_ai_research/data/independent_real/val2017/
via https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip (815MB).

Does NOT overwrite sdxl_index.csv / flux_index.csv. Writes:
  routing/results/sdxl_index_coco_real.csv
  routing/results/flux_index_coco_real.csv
  routing/results/sdxl_realonly_coco.csv   (real-only, for feature extraction)
  routing/results/flux_realonly_coco.csv
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_aigibench_index import find_images  # reuse, don't reimplement

random.seed(42)

ROUTING = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROUTING, "results")
COCO_DIR = "/workspace/benchmarks_ai_research/data/independent_real/val2017"
N_SAMPLES = 2000

EXISTING = [
    ("sdxl_index.csv", "sdxl_index_coco_real.csv", "sdxl_realonly_coco.csv"),
    ("flux_index.csv", "flux_index_coco_real.csv", "flux_realonly_coco.csv"),
]


def load_fake_rows(index_csv):
    with open(os.path.join(ROUTING, index_csv)) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["label"]) == 1]


def main():
    os.makedirs(RESULTS, exist_ok=True)

    print(f"Scanning COCO val2017 real images under {COCO_DIR} ...")
    real_imgs = find_images(COCO_DIR)
    print(f"  Found {len(real_imgs)} COCO real images")
    if len(real_imgs) == 0:
        print("ERROR: no COCO images found.")
        return

    for existing_csv, out_name, realonly_name in EXISTING:
        fake_rows = load_fake_rows(existing_csv)
        n = min(N_SAMPLES, len(real_imgs))
        real_sample = random.sample(real_imgs, n)  # continues global random
        # stream, same approach as build_original_real_index.py

        rows = (
            [{"path": p, "label": 0, "generator": "real_coco_val2017"} for p in real_sample] +
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
        print(f"  Saved {out_path} (real={n_real} [COCO], fake={n_fake} [existing {existing_csv} fakes], total={len(rows)})")

        realonly_path = os.path.join(RESULTS, realonly_name)
        with open(realonly_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
            writer.writeheader()
            writer.writerows([{"path": p, "label": 0, "generator": "real_coco_val2017"} for p in real_sample])
        print(f"  Saved {realonly_path} (real-only index, n={len(real_sample)})")


if __name__ == "__main__":
    main()
