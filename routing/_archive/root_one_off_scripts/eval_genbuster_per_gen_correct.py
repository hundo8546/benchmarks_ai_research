import csv
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, TimesformerForVideoClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16

def read_video(path, n=16):
    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr)-1, n).astype(int)
    return vr.get_batch(idx).asnumpy()

processor = AutoImageProcessor.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
)
model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
).to(DEVICE)
model.eval()

# Load index
with open("benchmark_index_with_gen.csv") as f:
    rows = list(csv.DictReader(f))

# Separate real and fake-by-generator
real_rows = [r for r in rows if r["label"] == "0"]
fake_by_gen = defaultdict(list)
for r in rows:
    if r["label"] == "1":
        fake_by_gen[r["generator"]].append(r)

# Precompute real scores once
real_scores = []
for r in tqdm(real_rows, desc="Scoring real videos"):
    frames = read_video(r["path"], NUM_FRAMES)
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        score = model(**inputs).logits.softmax(-1)[0, 1].item()
    real_scores.append(score)

print("\n===== Per-Generator Results (Real vs Fake) =====")

for gen, fake_rows in fake_by_gen.items():
    y_true = [0] * len(real_scores)
    y_score = real_scores.copy()

    for r in fake_rows:
        frames = read_video(r["path"], NUM_FRAMES)
        inputs = processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            score = model(**inputs).logits.softmax(-1)[0, 1].item()
        y_true.append(1)
        y_score.append(score)

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, [1 if s > 0.5 else 0 for s in y_score])

    print(f"{gen:>8s} | Fake={len(fake_rows):4d} | AUC={auc:.3f} | ACC={acc:.3f}")
