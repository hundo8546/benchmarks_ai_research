import time
import sys
import os

sys.path.append("/workspace/benchmarks_ai_research/CNNDetection")
import torch
import numpy as np
import pandas as pd
import csv
# from decord import VideoReader, cpu
# import cv2
from tqdm import tqdm
from networks.resnet import resnet50
from PIL import Image

DEVICE = "cuda"

MODEL_PATH = "/workspace/benchmarks_ai_research/weights/blur_jpg_prob0.1.pth"
INDEX_FILE = "/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv"
OUTPUT_CSV = "/workspace/benchmarks_ai_research/bandit_dataset.csv"

# INDEX_FILE = "/workspace/benchmarks_ai_research/routing/sd14_index.csv"
# OUTPUT_CSV = "/workspace/benchmarks_ai_research/routing/sd14_bandit_dataset.csv"

# =============================
# Load Model
# =============================
model = resnet50(num_classes=1)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict["model"])
model.to(DEVICE)
model.eval()

# =============================
# Frame Sampler
# =============================
def sample_frame(path):
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[0].asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.transpose(2, 0, 1)
    frame = frame / 255.0
    frame = torch.tensor(frame).float().unsqueeze(0)
    return frame.to(DEVICE)

# def sample_frame(path):
#     img = Image.open(path).convert("RGB")
#     img = img.resize((224, 224))
#     img = np.array(img).transpose(2, 0, 1)
#     img = img / 255.0
#     return torch.tensor(img).float().unsqueeze(0).to(DEVICE)

# =============================
# Process Dataset
# =============================
rows_out = []

with open(INDEX_FILE) as f:
    rows = list(csv.DictReader(f))
    
for row in tqdm(rows):
    path = row["path"]
    label = int(row["label"])
    gen = row["generator"]

    img = sample_frame(path)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = model(img)
        logit = output.item()
        prob_fake = torch.sigmoid(output).item()

    torch.cuda.synchronize()
    end = time.perf_counter()

    latency_cnn_ms = (end - start) * 1000

    prob_real = 1 - prob_fake
    confidence = max(prob_fake, prob_real)
    margin = abs(logit)

    eps = 1e-12
    entropy = -(
        prob_fake * np.log(prob_fake + eps) +
        prob_real * np.log(prob_real + eps)
    )

    pred = 1 if prob_fake > 0.5 else 0

    rows_out.append({
        "path": path,
        "logit": logit,
        "confidence": confidence,
        "margin1": margin,
        "entropy1": entropy,
        "y_cnn": pred,
        "y_true": label,
        "generator": gen,
        "latency_cnn_ms": latency_cnn_ms
    })

df = pd.DataFrame(rows_out)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved bandit_dataset.csv with", len(df), "rows")