import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
import time, torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
t0 = time.time()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True,
).eval()
print("load time", time.time() - t0, flush=True)
print("mem allocated GB:", torch.cuda.memory_allocated() / 1e9, flush=True)

from PIL import Image
import pandas as pd
df = pd.read_csv("sdxl_gpt55_siglip2_disagree_preds.csv").head(3)
PROMPT = ("Analyze this image carefully. Determine whether it is a REAL photograph "
          "or an AI-generated FAKE image. Consider both possibilities equally.\nAnswer with exactly one word: REAL or FAKE.")
real_id = processor.tokenizer.encode("REAL", add_special_tokens=False)[0]
fake_id = processor.tokenizer.encode("FAKE", add_special_tokens=False)[0]
for _, r in df.iterrows():
    img = Image.open(r["path"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False, return_dict_in_generate=True, output_scores=True)
    dt = time.time() - t1
    new_toks = gen.sequences[:, inputs["input_ids"].shape[-1]:]
    raw = processor.batch_decode(new_toks, skip_special_tokens=True)[0].strip()
    print(r["path"], "y_true=", r["y_true"], "raw=", raw, "latency_s=", round(dt, 2), flush=True)

print("SMOKE TEST DONE", flush=True)
