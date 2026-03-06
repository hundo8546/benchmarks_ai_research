import csv
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
OUT_FILE = "qwen_preds.csv"

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

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

rows = []
with open(INDEX_FILE) as f:
    index_rows = list(csv.DictReader(f))

prompt = (
    "You are a multimedia forensics assistant. "
    "Classify this video frame as REAL (camera-captured) or FAKE (AI-generated). "
    "Answer with exactly one word: REAL or FAKE."
)

for r in tqdm(index_rows):
    path = r["path"]
    img = sample_frame_pil(path)

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


    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    latency_qwen_ms = (end - start) * 1000
    

    # Remove prompt tokens
    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]

    output_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()

    y = parse_yes_no(output_text)
    if y == -1:
        y = 1  # fallback

    #print(output_text)
    rows.append({
        "path": path,
        "y_qwen": y,
        "raw": output_text,
        "latency_qwen_ms": latency_qwen_ms,

    })

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print("Wrote", OUT_FILE)