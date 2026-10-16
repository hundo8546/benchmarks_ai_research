"""
Test whether the shorter e4_strong_verifier.py-style prompt changes
Qwen2.5-VL-7B's real/fake bias on the same 200-image BigGAN distant-shift
subsample, compared to the biggan_run_qwen_v2.py-style prompt already used.
"""
import os
import time

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda"

ALT_PROMPT = (
    "Determine whether this image is real (camera-captured) or AI-generated.\n"
    "Answer with exactly one word: REAL or FAKE."
)


def parse_label(text):
    t = text.strip().upper()
    has_fake = "FAKE" in t
    has_real = "REAL" in t
    if has_fake and not has_real: return 1
    if has_real and not has_fake: return 0
    return None


def main():
    df = pd.read_csv(os.path.join(ROUTING_BASE, "prompt_test_subsample.csv"))
    print("Loading Qwen2.5-VL-7B-Instruct...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    ).eval()

    rows = []
    for r in tqdm(df.itertuples(), total=len(df)):
        img = Image.open(r.path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ALT_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        new_toks = gen[:, inputs["input_ids"].shape[-1]:]
        raw = processor.batch_decode(new_toks, skip_special_tokens=True)[0].strip()
        y = parse_label(raw)
        rows.append({"path": r.path, "y_true": int(r.y_true), "y_qwen_altprompt": y, "raw": raw})
        del inputs, gen, new_toks

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROUTING_BASE, "prompt_test_altprompt_results.csv"), index=False)

    valid = out.dropna(subset=["y_qwen_altprompt"])
    acc = (valid["y_qwen_altprompt"] == valid["y_true"]).mean()
    print(f"\nAlt-prompt accuracy: {acc:.3f} (n={len(valid)}/{len(out)} parsed)")
    for t in [0, 1]:
        sub = valid[valid.y_true == t]
        print(f"  y_true={t}: correct rate = {(sub.y_qwen_altprompt==sub.y_true).mean():.3f} (n={len(sub)})")
    print(f"  predicted distribution: {valid['y_qwen_altprompt'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
