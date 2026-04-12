import sys
import os
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from decord import VideoReader, cpu
# import torchvision.transforms as T
# from PIL import ImageFilter
import cv2
from networks.resnet import resnet50

DEVICE = "cuda"
MODEL_PATH = "/workspace/benchmarks_ai_research/CNNDetection_official/weights/blur_jpg_prob0.5.pth"
INDEX_FILE = "/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv"
OUT_FILE = "bandit_dataset.csv"

# Load CNNSpot
model = resnet50(num_classes=1)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# def blur_pil(img):
#     return img.filter(ImageFilter.GaussianBlur(radius=2))

def sample_frame(path):
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[0].asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.transpose(2, 0, 1)
    frame = frame / 255.0
    frame = torch.tensor(frame).float().unsqueeze(0)
    return frame.to(DEVICE)

rows_out = []

with open(INDEX_FILE) as f:
    rows = list(csv.DictReader(f))

for row in tqdm(rows):
    path = row["path"]
    label = int(row["label"])
    gen = row["generator"]

    # ----- ORIGINAL FRAME -----
    img = sample_frame(path)

    with torch.no_grad():
        output = model(img)
        logit = output.item()
        prob_fake = torch.sigmoid(output).item()
        prob_real = 1 - prob_fake

    pred_orig = 1 if prob_fake > 0.5 else 0

    confidence = max(prob_fake, prob_real)
    margin = abs(logit)

    eps = 1e-12
    entropy = -(
        prob_fake * np.log(prob_fake + eps) +
        prob_real * np.log(prob_real + eps)
    )

    # ----- BLUR STABILITY -----
    # reload frame as image for blur
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[0].asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_blur = cv2.GaussianBlur(frame, (5,5), 0)

    frame_blur = cv2.resize(frame_blur, (224, 224))
    frame_blur = frame_blur.transpose(2,0,1) / 255.0
    frame_blur = torch.tensor(frame_blur).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_blur = model(frame_blur)
        prob_fake_blur = torch.sigmoid(out_blur).item()

    pred_blur = 1 if prob_fake_blur > 0.5 else 0

    flip = int(pred_orig != pred_blur)

    rows_out.append({
        "path": path,
        "logit": logit,
        "confidence": confidence,
        "margin1": margin,
        "entropy1": entropy,
        "flip": flip,
        "y_cnn": pred_orig,
        "y_true": label,
        "generator": gen
    })

df = pd.DataFrame(rows_out)
df.to_csv(OUT_FILE, index=False)

print("Saved", OUT_FILE, "with", len(df), "rows")