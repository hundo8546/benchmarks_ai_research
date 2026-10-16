"""
Build the GenImage BigGAN index CSV (real/fake path pairs, seeded sample).

Minimally refactored (routing/pipeline.py consolidation pass) into an
importable main() so routing/pipeline.py's --build-indices subcommand can
call it directly -- logic below is byte-for-byte unchanged from before the
refactor, just moved inside a function with the same module-level constants.
"""
import os
import random
import csv
from pathlib import Path

VAL_ROOT = "/workspace/benchmarks_ai_research/data/genimage/biggan/imagenet_ai_0419_biggan/val"
OUT_INDEX = "/workspace/benchmarks_ai_research/routing/biggan_index.csv"


# Get all images
def get_images(directory):
    images = []
    for ext in ["*.jpg", "*.JPEG", "*.png", "*.jpeg"]:
        images += [str(p) for p in Path(directory).rglob(ext)]
    return images


def main():
    random.seed(42)

    real_dir = os.path.join(VAL_ROOT, "nature")
    fake_dir = os.path.join(VAL_ROOT, "ai")

    real_images = get_images(real_dir)
    fake_images = get_images(fake_dir)

    print(f"Found {len(real_images)} real images")
    print(f"Found {len(fake_images)} fake images")

    # Build index
    rows = []
    for path in real_images:
        rows.append({"path": path, "label": 0, "generator": "real"})
    for path in fake_images:
        rows.append({"path": path, "label": 1, "generator": "biggan"})

    random.shuffle(rows)

    with open(OUT_INDEX, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUT_INDEX}")
    print(f"Real: {sum(1 for r in rows if r['label']==0)}")
    print(f"Fake: {sum(1 for r in rows if r['label']==1)}")
    return rows


if __name__ == "__main__":
    main()