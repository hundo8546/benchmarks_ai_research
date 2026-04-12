import os
import random
import csv
from pathlib import Path

random.seed(42)

VAL_ROOT = "/workspace/benchmarks_ai_research/data/genimage/sd_1_4/data/genimage/sd_1_4/imagenet_ai_0419_sdv4/val"
OUT_INDEX = "/workspace/benchmarks_ai_research/routing/sd14_index.csv"

real_dir = os.path.join(VAL_ROOT, "nature")
fake_dir = os.path.join(VAL_ROOT, "ai")

# Get all images
def get_images(directory):
    images = []
    for ext in ["*.jpg", "*.JPEG", "*.png", "*.jpeg"]:
        images += [str(p) for p in Path(directory).rglob(ext)]
    return images

real_images = get_images(real_dir)
fake_images = get_images(fake_dir)

print(f"Found {len(real_images)} real images")
print(f"Found {len(fake_images)} fake images")

# Build index
rows = []
for path in real_images:
    rows.append({"path": path, "label": 0, "generator": "real"})
for path in fake_images:
    rows.append({"path": path, "label": 1, "generator": "sdv4"})

random.shuffle(rows)

with open(OUT_INDEX, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUT_INDEX}")
print(f"Real: {sum(1 for r in rows if r['label']==0)}")
print(f"Fake: {sum(1 for r in rows if r['label']==1)}")