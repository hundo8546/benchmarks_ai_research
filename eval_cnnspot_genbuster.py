import sys
sys.path.insert(0, "/workspace/CNNDetection")

import torch
import numpy as np
import csv
import cv2
from tqdm import tqdm
from decord import VideoReader, cpu

from networks.resnet import resnet50

DEVICE = "cuda"
MODEL_PATH = "/workspace/CNNDetection/weights/blur_jpg_prob0.5.pth"
INDEX_FILE = "/workspace/benchmark_index_with_gen.csv"

# Load model
model = resnet50(num_classes=1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

def sample_frame(path):
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[0].asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.transpose(2, 0, 1)
    frame = frame / 255.0
    frame = torch.tensor(frame).float().unsqueeze(0)
    return frame.to(DEVICE)

results = {"real": [], "fake": []}
per_gen = {}

with open(INDEX_FILE) as f:
    rows = list(csv.DictReader(f))

for row in tqdm(rows):
    path = row["path"]
    label = int(row["label"])
    gen = row["generator"]

    img = sample_frame(path)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    pred = 1 if prob > 0.5 else 0
    correct = (pred == label)

    if label == 0:
        results["real"].append(correct)
    else:
        results["fake"].append(correct)
        if gen not in per_gen:
            per_gen[gen] = []
        per_gen[gen].append(correct)

real_acc = np.mean(results["real"])
fake_acc = np.mean(results["fake"])
overall = np.mean([real_acc, fake_acc])

print("\n===== CNNSpot RESULTS =====")
print("Real Acc:", real_acc)
print("Fake Acc:", fake_acc)
print("Overall:", overall)

print("\n===== Per-Generator Fake Accuracy =====")
for gen in per_gen:
    print(gen, np.mean(per_gen[gen]))