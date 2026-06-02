"""
Build index CSVs for AIGIBench SDXL and FLUX subsets.

Walks the extracted directories, pairs real images with AI-generated images,
samples down to --n_samples per class (default 2000 each), and writes:
  routing/sdxl_index.csv
  routing/flux_index.csv

Usage:
    python build_aigibench_index.py
    python build_aigibench_index.py --n_samples 3000
"""
import argparse
import csv
import os
import random
from pathlib import Path

random.seed(42)

BASE       = "/workspace/benchmarks_ai_research/data/aigibench"
# Use ImageNet val images as real — same distribution as SD14 probe training data.
# AIGIBench PASCAL VOC real images are too old/low-res; GPT-5.5 misclassifies them.
REAL_DIR   = "/workspace/benchmarks_ai_research/data/genimage/sd_1_4/imagenet_ai_0419_sdv4/val/nature"
SDXL_DIR   = os.path.join(BASE, "sdxl")
FLUX_DIR   = os.path.join(BASE, "flux")
ROUTING    = os.path.join(os.path.dirname(__file__))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG"}


def find_images(root, only_subdir=None):
    """Find images recursively. If only_subdir given, only include paths
    whose parent directory name matches (e.g. '0_real' or '1_fake')."""
    imgs = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if p.name.startswith("."):
            continue
        if only_subdir and p.parent.name != only_subdir:
            continue
        imgs.append(str(p))
    return imgs


def build_index(real_imgs, fake_imgs, n_samples, generator_tag, out_path):
    real_sample = random.sample(real_imgs, min(n_samples, len(real_imgs)))
    fake_sample = random.sample(fake_imgs, min(n_samples, len(fake_imgs)))

    rows = (
        [{"path": p, "label": 0, "generator": "real"} for p in real_sample] +
        [{"path": p, "label": 1, "generator": generator_tag} for p in fake_sample]
    )
    random.shuffle(rows)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
        writer.writeheader()
        writer.writerows(rows)

    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)
    print(f"  Saved {out_path}  (real={n_real}, fake={n_fake}, total={len(rows)})")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=2000,
                   help="Samples per class (real and fake)")
    args = p.parse_args()

    if not os.path.exists(REAL_DIR):
        print(f"ERROR: real images not found at {REAL_DIR}")
        return

    print("Scanning real images (ImageNet val/nature) ...")
    real_imgs = find_images(REAL_DIR)
    print(f"  Found {len(real_imgs)} real images")

    if len(real_imgs) == 0:
        print("No real images found — check extraction.")
        return

    for generator, gen_dir, out_name in [
        ("sdxl",      SDXL_DIR, "sdxl_index.csv"),
        ("flux1-dev", FLUX_DIR, "flux_index.csv"),
    ]:
        gen_dir_exists = os.path.exists(gen_dir)
        if not gen_dir_exists:
            print(f"\nSkipping {generator}: {gen_dir} not found")
            continue

        print(f"\nScanning {generator} images (1_fake/ subdirs only) ...")
        fake_imgs = find_images(gen_dir, only_subdir="1_fake")
        if not fake_imgs:
            fake_imgs = find_images(gen_dir)
        print(f"  Found {len(fake_imgs)} fake images")

        if len(fake_imgs) == 0:
            print(f"  No images found — check extraction of {generator}")
            continue

        out_path = os.path.join(ROUTING, out_name)
        print(f"Building {generator} index (n_samples={args.n_samples} per class) ...")
        build_index(real_imgs, fake_imgs, args.n_samples, generator, out_path)

    print("\nDone. Next:")
    print("  python experiments/e8_modern_diffusion.py --extract_features")


if __name__ == "__main__":
    main()
