import csv
import torch
import pandas as pd
from tqdm import tqdm
import time
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

torch.cuda.empty_cache()

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INDEX_FILE = "/workspace/benchmarks_ai_research/routing/biggan_index.csv"
OUT_FILE = "/workspace/benchmarks_ai_research/routing/biggan_qwen_preds.csv"

# -----------------------------
# Image loader (resize!!)
# -----------------------------
def sample_frame_pil(path):
    return Image.open(path).convert("RGB").resize((224, 224))


def parse_yes_no(text: str) -> int:
    t = text.strip().upper()
    if "REAL" in t:
        return 0
    if "FAKE" in t:
        return 1
    return 1  # fallback


# -----------------------------
# Load model (4-bit!!)
# -----------------------------
print("Loading processor and model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
).eval()


# -----------------------------
# Load data
# -----------------------------
rows = []

with open(INDEX_FILE) as f:
    index_rows = list(csv.DictReader(f))

# keep it small for debugging
index_rows = index_rows[:20]

# -----------------------------
# Prompt
# -----------------------------
PROMPT = (
    "Analyze this image. Is it REAL (photograph) or FAKE (AI-generated)? "
    "Answer with one word: REAL or FAKE."
)

# -----------------------------
# Inference loop
# -----------------------------
for i, r in enumerate(tqdm(index_rows)):
    path = r["path"]

    try:
        img = sample_frame_pil(path)
    except:
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
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
    )

    start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            use_cache=False,
        )

    end = time.perf_counter()
    latency = (end - start) * 1000

    new_tokens = generated_ids[:, inputs["input_ids"].shape[-1]:]

    output_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()

    y = parse_yes_no(output_text)

    print(f"[{i}] {output_text}")

    rows.append({
        "path": path,
        "y_qwen": y,
        "raw": output_text,
        "latency_qwen_ms": latency,
    })


# -----------------------------
# Save
# -----------------------------
pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print("Wrote", OUT_FILE)