import csv
import gc
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from decord import VideoReader, cpu
import cv2
import time
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
torch.cuda.empty_cache()

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INDEX_FILE = "/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv"
VAL_PATHS_NPY = "/workspace/benchmarks_ai_research/routing/bandit_val_paths.npy"
OUT_FILE = "/workspace/benchmarks_ai_research/routing/qwen_preds.csv"


def sample_frame_pil(path):
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[0].asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def parse_yes_no(text: str) -> int:
    t = text.strip().upper()
    if t == "REAL":
        return 0
    if t == "FAKE":
        return 1
    return -1


# Load only the val paths so we run on the correct split
val_paths = set(np.load(VAL_PATHS_NPY, allow_pickle=True).tolist())

with open(INDEX_FILE) as f:
    index_rows = [r for r in csv.DictReader(f) if r["path"] in val_paths]

print(f"Running Qwen on {len(index_rows)} GenBuster val samples...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

rows = []
for r in tqdm(index_rows):
    path = r["path"]
    try:
        img = sample_frame_pil(path)
    except Exception as e:
        print(f"Skip {path}: {e}")
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text":
                    "Classify this image as REAL or FAKE. "
                    "Answer with exactly one word: REAL or FAKE."
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    ).to(DEVICE)
    del img

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )
    torch.cuda.synchronize()
    latency_qwen_ms = (time.perf_counter() - start) * 1000

    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]
    del inputs, generated_ids

    output_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()

    y = parse_yes_no(output_text)
    if y == -1:
        y = 1  # fallback

    rows.append({
        "path": path,
        "y_true": int(r["label"]),
        "y_qwen": y,
        "y_qwen_text": output_text,
        "latency_qwen_ms": latency_qwen_ms,
    })

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"Wrote {len(rows)} rows to {OUT_FILE}")