"""
Run Qwen2.5-VL-7B-Instruct once, locally, on two queues that both need new
predictions:
  1. GenBuster: 799 new val-set videos not covered by the old 400-sample
     qwen_preds.csv (genbuster_qwen_todo.csv).
  2. BigGAN distant-shift: the 1,500-sample stratified disagreement subset
     (biggan_distant_shift_1500.csv) where SD1.4-stale FFT+SigLIP2 disagree.

Reuses the same model-loading / inference / label-parsing logic as
routing/biggan_run_qwen_v2.py, generalized for video vs static images.
"""
import gc
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REAL_LABEL, FAKE_LABEL, UNK_LABEL = 0, 1, -1

REASONING_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL photograph "
    "or an AI-generated FAKE image. Consider both possibilities equally.\n"
    "Answer with exactly one word: REAL or FAKE."
)


def clear_gpu():
    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()


def sample_frame_pil(path, is_video):
    if is_video:
        vr = VideoReader(path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return Image.open(path).convert("RGB")


def parse_label(text):
    if not text:
        return UNK_LABEL
    lines = [ln.strip().upper() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if last == "REAL": return REAL_LABEL
        if last == "FAKE": return FAKE_LABEL
        if re.search(r"\bFAKE\b", last) and not re.search(r"\bREAL\b", last): return FAKE_LABEL
        if re.search(r"\bREAL\b", last) and not re.search(r"\bFAKE\b", last): return REAL_LABEL
    matches = list(re.finditer(r"\b(REAL|FAKE)\b", text.upper()))
    if matches:
        return REAL_LABEL if matches[-1].group(1) == "REAL" else FAKE_LABEL
    return UNK_LABEL


def load_model():
    print("Loading Qwen2.5-VL-7B-Instruct...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    clear_gpu()
    return processor, model


def infer(model, processor, img):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": REASONING_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 1000
    new_toks = gen[:, inputs["input_ids"].shape[-1]:]
    raw = processor.batch_decode(new_toks, skip_special_tokens=True)[0].strip()
    y = parse_label(raw)
    del inputs, gen, new_toks
    return y, raw, lat


def run_queue(model, processor, todo_csv, out_csv, is_video, checkpoint_every=25):
    df = pd.read_csv(todo_csv)
    done_rows, done_paths = [], set()
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        done_rows = df_old.to_dict("records")
        done_paths = set(df_old["path"])
    todo = df[~df["path"].isin(done_paths)].reset_index(drop=True)
    print(f"\n=== {out_csv} === total={len(df)} already_done={len(done_paths)} todo={len(todo)}")

    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo))):
        try:
            img = sample_frame_pil(r.path, is_video)
            y, raw, lat = infer(model, processor, img)
        except Exception as ex:
            y, raw, lat = FAKE_LABEL, f"ERROR:{ex}", 0.0
        done_rows.append({"path": r.path, "y_true": int(r.y_true), "y_qwen": y,
                          "raw": raw, "latency_qwen_ms": lat})
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(done_rows).to_csv(out_csv, index=False)
            recent = done_rows[-checkpoint_every:]
            n_fake = sum(1 for x in recent if x["y_qwen"] == FAKE_LABEL)
            print(f"  [checkpoint {i+1}/{len(todo)}] last {checkpoint_every}: "
                  f"{n_fake} FAKE / {checkpoint_every - n_fake} REAL")
        if (i + 1) % 5 == 0:
            clear_gpu()

    pd.DataFrame(done_rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(done_rows)} rows)")


def main():
    processor, model = load_model()
    START = time.time()
    run_queue(model, processor,
               os.path.join(ROUTING_BASE, "genbuster_qwen_todo.csv"),
               os.path.join(ROUTING_BASE, "genbuster_qwen_expansion_preds.csv"),
               is_video=True)
    print(f"GenBuster done at +{time.time()-START:.0f}s")
    run_queue(model, processor,
               os.path.join(ROUTING_BASE, "biggan_distant_shift_1500.csv"),
               os.path.join(ROUTING_BASE, "biggan_distant_shift_qwen7b_preds.csv"),
               is_video=False)
    print(f"BigGAN distant-shift done at +{time.time()-START:.0f}s")


if __name__ == "__main__":
    main()
