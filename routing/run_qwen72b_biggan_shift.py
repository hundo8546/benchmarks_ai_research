"""
Run Qwen2.5-VL-72B-Instruct (bnb-4bit) as the third verifier on the BigGAN
distant-shift 1,500-sample subset, for comparability with the already-run
GPT-5.5 and Qwen2.5-VL-7B verifiers (biggan_distant_shift_gpt55_preds.csv,
biggan_distant_shift_qwen7b_preds.csv).

This pod has a single 96GB-VRAM GPU, so the ~42GB quantized model fits
entirely on-device -- no CPU offload needed.
"""
import gc
import os
import re
import sys
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"
MODEL_ID = "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit"
REAL_LABEL, FAKE_LABEL, UNK_LABEL = 0, 1, -1

TODO_CSV = os.path.join(ROUTING_BASE, "biggan_distant_shift_1500.csv")
OUT_CSV = os.path.join(ROUTING_BASE, "biggan_distant_shift_qwen72b_preds.csv")

REASONING_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL photograph "
    "or an AI-generated FAKE image. Consider both possibilities equally.\n"
    "Answer with exactly one word: REAL or FAKE."
)


def clear_gpu():
    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()


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
    print(f"Loading {MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="cuda:0", low_cpu_mem_usage=True,
    ).eval()
    clear_gpu()
    print(f"Model loaded in {time.time()-t0:.0f}s. "
          f"GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return processor, model


def infer(model, processor, img):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": REASONING_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 1000
    new_toks = gen[:, inputs["input_ids"].shape[-1]:]
    raw = processor.batch_decode(new_toks, skip_special_tokens=True)[0].strip()
    y = parse_label(raw)
    del inputs, gen, new_toks
    return y, raw, lat


def smoke_test(model, processor, df, n=3):
    print(f"\n=== Smoke test on {n} images ===")
    for r in df.head(n).itertuples():
        img = Image.open(r.path).convert("RGB")
        y, raw, lat = infer(model, processor, img)
        print(f"  {r.path} true={r.y_true} pred={y} raw={raw!r} latency={lat:.0f}ms")
    clear_gpu()


def run(model, processor, checkpoint_every=25):
    df = pd.read_csv(TODO_CSV)
    done_rows, done_paths = [], set()
    if os.path.exists(OUT_CSV):
        df_old = pd.read_csv(OUT_CSV)
        done_rows = df_old.to_dict("records")
        done_paths = set(df_old["path"])
    todo = df[~df["path"].isin(done_paths)].reset_index(drop=True)
    print(f"\n=== {OUT_CSV} === total={len(df)} already_done={len(done_paths)} todo={len(todo)}")

    start = time.time()
    for i, r in enumerate(todo.itertuples()):
        try:
            img = Image.open(r.path).convert("RGB")
            y, raw, lat = infer(model, processor, img)
        except Exception as ex:
            y, raw, lat = FAKE_LABEL, f"ERROR:{ex}", 0.0
        done_rows.append({"path": r.path, "y_true": int(r.y_true), "y_qwen": y,
                          "raw": raw, "latency_qwen_ms": lat})
        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(todo):
            pd.DataFrame(done_rows).to_csv(OUT_CSV, index=False)
            recent = done_rows[-checkpoint_every:]
            n_fake = sum(1 for x in recent if x["y_qwen"] == FAKE_LABEL)
            n_real = sum(1 for x in recent if x["y_qwen"] == REAL_LABEL)
            n_unk = len(recent) - n_fake - n_real
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(todo) - (i + 1)) / rate if rate > 0 else float("inf")
            print(f"  [checkpoint {i+1}/{len(todo)}] last {len(recent)}: "
                  f"{n_fake} FAKE / {n_real} REAL / {n_unk} UNK | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")
            sys.stdout.flush()
        if (i + 1) % 5 == 0:
            clear_gpu()

    pd.DataFrame(done_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} ({len(done_rows)} rows)")


def main():
    processor, model = load_model()
    df = pd.read_csv(TODO_CSV)
    smoke_test(model, processor, df, n=3)
    run(model, processor)


if __name__ == "__main__":
    main()
