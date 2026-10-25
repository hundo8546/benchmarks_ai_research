import csv
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, TimesformerForVideoClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16   # safe for RTX A4500 (20GB)
BATCH = 1

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

y_true, y_score = [], []

with open("benchmark_index.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for r in tqdm(rows):
    frames = read_video(r["path"], NUM_FRAMES)
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        score = logits.softmax(-1)[0, 1].item()

    y_true.append(int(r["label"]))
    y_score.append(score)

auc = roc_auc_score(y_true, y_score)
pred = [1 if s > 0.5 else 0 for s in y_score]
acc = accuracy_score(y_true, pred)

print("========== RESULTS ==========")
print(f"Samples: {len(y_true)}")
print(f"ROC-AUC: {auc:.4f}")
print(f"ACC:     {acc:.4f}")
