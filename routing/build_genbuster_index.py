"""
Rebuild benchmark_index_with_gen.csv from the actual extracted GenBuster data.
Samples up to MAX_PER_CLASS real and fake videos.
"""
import csv
import random
from pathlib import Path

random.seed(42)

REAL_DIR  = Path("/workspace/data/genbuster-mini/extracted/train/real")
FAKE_DIR  = Path("/workspace/data/genbuster-mini/extracted/train/fake")
OUT_INDEX = "/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv"

MAX_PER_CLASS = 1000   # 1k real + 1k fake = 2k total, matching original scale

real_files = list(REAL_DIR.glob("*.mp4"))
fake_files = []
for gen_dir in FAKE_DIR.iterdir():
    if gen_dir.is_dir():
        fake_files.extend(gen_dir.glob("*.mp4"))

print(f"Found {len(real_files)} real, {len(fake_files)} fake")

real_sample = random.sample(real_files, min(MAX_PER_CLASS, len(real_files)))
fake_sample = random.sample(fake_files, min(MAX_PER_CLASS, len(fake_files)))

rows = []
for p in real_sample:
    rows.append({"path": str(p), "label": 0, "generator": "real"})
for p in fake_sample:
    gen = p.parent.name
    rows.append({"path": str(p), "label": 1, "generator": gen})

random.shuffle(rows)

with open(OUT_INDEX, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "label", "generator"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUT_INDEX}")
print(f"Real: {sum(1 for r in rows if r['label'] == 0)}")
print(f"Fake: {sum(1 for r in rows if r['label'] == 1)}")
print(f"Generators: {sorted(set(r['generator'] for r in rows))}")
