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

data = defaultdict(lambda: {"y": [], "s": []})

with open("benchmark_index_with_gen.csv") as f:
    rows = list(csv.DictReader(f))

for r in tqdm(rows):
    frames = read_video(r["path"], NUM_FRAMES)
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        score = logits.softmax(-1)[0, 1].item()

    gen = r["generator"]
    data[gen]["y"].append(int(r["label"]))
    data[gen]["s"].append(score)

print("\n===== Per-Generator Results =====")
for gen, d in sorted(data.items()):
    if len(set(d["y"])) < 2:
        continue
    auc = roc_auc_score(d["y"], d["s"])
    acc = accuracy_score(d["y"], [1 if s > 0.5 else 0 for s in d["s"]])
    print(f"{gen:>8s} | N={len(d['y']):4d} | AUC={auc:.3f} | ACC={acc:.3f}")
