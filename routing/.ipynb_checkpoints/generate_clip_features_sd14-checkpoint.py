import os
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"
import csv
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
#from decord import VideoReader, cpu
#import cv2
from PIL import Image
import time

from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda"
MODEL_ID = "openai/clip-vit-base-patch32"
# INDEX_FILE = "/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv"
# OUT_FILE = "/workspace/benchmarks_ai_research/routing/clip_features.csv"

INDEX_FILE = "/workspace/benchmarks_ai_research/routing/sd14_index.csv"
OUT_FILE = "/workspace/benchmarks_ai_research/routing/sd14_clip_features.csv"




processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# def sample_frame_pil(path):
#     vr = VideoReader(path, ctx=cpu(0))
#     frame = vr[0].asnumpy()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(frame)

def sample_frame_pil(path):
    img = Image.open(path).convert("RGB")
    return img

rows = []

with open(INDEX_FILE) as f:
    index_rows = list(csv.DictReader(f))

for r in tqdm(index_rows):
    path = r["path"]
    label = int(r["label"])
    generator = r["generator"]

    img = sample_frame_pil(path)

    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    

    torch.cuda.synchronize()
    end = time.perf_counter()
    latency_clip_ms = (end - start) * 1000
    
    image_features = model.get_image_features(**inputs)[0].detach().cpu().numpy().flatten()

    rows.append({
        "path": path,
        "y_true": label,
        **{f"f{i}": image_features[i] for i in range(len(image_features))},
        "latency_clip_ms": latency_clip_ms,
        "generator": generator
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)

print("Saved", OUT_FILE)