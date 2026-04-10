import csv
import os
import re
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
import cv2
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText

# ============================================
# CONFIGURATION
# ============================================
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Dataset specific settings
INDEX_FILE = "/workspace/benchmarks_ai_research/routing/biggan_index.csv"
OUT_FILE = "biggan_qwen_preds.csv"
IS_VIDEO = False  # True for GenBuster, False for SD14/BigGAN

# Test settings
SMOKE_TEST_N = None  # Set to None for full dataset

# Memory management settings
CLEAR_CACHE_EVERY_N = 5  # Clear GPU cache every N samples
MAX_RETRIES_ON_OOM = 3   # Retry failed samples up to 3 times
BATCH_SIZE = 1           # Keep at 1 for stability

# Labels
REAL_LABEL = 0
FAKE_LABEL = 1
UNK_LABEL = -1

# Prompt (neutral version to reduce bias)
REASONING_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL photograph "
    "or an AI-generated FAKE image. Consider both possibilities equally.\n"
    "Answer with exactly one word: REAL or FAKE."
)

# ============================================
# MEMORY MANAGEMENT FUNCTIONS
# ============================================
def clear_gpu_memory(verbose=False):
    """Aggressively clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def print_gpu_memory(stage=""):
    """Debug function to print GPU memory usage"""
    if torch.cuda.is_available() and stage:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

def cleanup_model(model, processor):
    """Properly delete model and processor to free memory"""
    if model is not None:
        del model
    if processor is not None:
        del processor
    clear_gpu_memory(verbose=True)

# ============================================
# IMAGE LOADING
# ============================================
def sample_frame_pil(path: str) -> Image.Image:
    """Single frame sampler for both videos and static images"""
    if IS_VIDEO:
        vr = VideoReader(path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    else:
        return Image.open(path).convert("RGB")

# ============================================
# LABEL PARSING
# ============================================
def parse_label(text: str) -> int:
    """Robust label parser"""
    if not text:
        return UNK_LABEL
    
    lines = [ln.strip().upper() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if last == "REAL":
            return REAL_LABEL
        if last == "FAKE":
            return FAKE_LABEL
        if re.search(r"\bFAKE\b", last) and not re.search(r"\bREAL\b", last):
            return FAKE_LABEL
        if re.search(r"\bREAL\b", last) and not re.search(r"\bFAKE\b", last):
            return REAL_LABEL
    
    matches = list(re.finditer(r"\b(REAL|FAKE)\b", text.upper()))
    if matches:
        last_match = matches[-1].group(1)
        return REAL_LABEL if last_match == "REAL" else FAKE_LABEL
    
    return UNK_LABEL

# ============================================
# SOFTMAX UTILITIES
# ============================================
def safe_softmax_two(a: float, b: float):
    """Safe softmax for two values"""
    x = np.array([a, b], dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / e.sum()
    return float(p[0]), float(p[1])

# ============================================
# TOKEN ID FINDING
# ============================================
def find_label_token_ids(processor):
    """Find token IDs for REAL and FAKE"""
    def first_token(text):
        ids = processor.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if ids else None
    
    real_id = first_token("REAL")
    fake_id = first_token("FAKE")
    
    print(f"\n=== Token IDs ===")
    print(f"REAL token ID: {real_id}")
    print(f"FAKE token ID: {fake_id}")
    print(f"REAL decoded: {processor.tokenizer.decode([real_id]) if real_id else 'N/A'}")
    print(f"FAKE decoded: {processor.tokenizer.decode([fake_id]) if fake_id else 'N/A'}")
    print("================\n")
    
    return real_id, fake_id

# ============================================
# MODEL LOADING (WITH MEMORY OPTIMIZATION)
# ============================================
def load_model():
    """Load model with memory optimizations"""
    print("Loading processor and model...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,  # Reduce CPU memory usage
    ).eval()
    
    # Enable memory efficient attention if available
    if hasattr(model, "config"):
        model.config.use_cache = True
    
    clear_gpu_memory(verbose=True)
    
    return processor, model

# ============================================
# SINGLE SAMPLE INFERENCE (WITH MEMORY MANAGEMENT)
# ============================================
def infer_single_sample(model, processor, img, retry_count=0):
    """Run inference on a single sample with retry logic"""
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": REASONING_PROMPT},
            ],
        }]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
        ).to(DEVICE)
        
        n_input_tokens = int(inputs["input_ids"].shape[-1])
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        
        new_tokens = generated.sequences[:, inputs["input_ids"].shape[-1]:]
        n_output_tokens = int(new_tokens.shape[-1])
        output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        
        y_text = parse_label(output_text)
        
        # Extract scores from logits
        p_fake = None
        p_real = None
        margin = None
        
        if generated.scores is not None and len(generated.scores) > 0:
            best_gap = -1.0
            best_p_real = None
            best_p_fake = None
            for step_logits in generated.scores:
                real_l = float(step_logits[0, real_token_id].item())
                fake_l = float(step_logits[0, fake_token_id].item())
                pr, pf = safe_softmax_two(real_l, fake_l)
                gap = abs(pf - pr)
                if gap > best_gap:
                    best_gap = gap
                    best_p_real = pr
                    best_p_fake = pf
            p_real = best_p_real
            p_fake = best_p_fake
            margin = best_gap
        
        # Determine final prediction
        if p_fake is not None:
            y_score = FAKE_LABEL if p_fake >= 0.5 else REAL_LABEL
        else:
            y_score = y_text if y_text != UNK_LABEL else FAKE_LABEL
        
        # Clean up intermediate tensors
        del inputs, generated, new_tokens
        
        return {
            "success": True,
            "y_score": y_score,
            "y_text": y_text,
            "p_fake": p_fake,
            "p_real": p_real,
            "margin": margin,
            "input_tokens": n_input_tokens,
            "output_tokens": n_output_tokens,
            "latency_ms": latency_ms,
            "raw_output": output_text,
        }
        
    except torch.cuda.OutOfMemoryError as e:
        if retry_count < MAX_RETRIES_ON_OOM:
            print(f"  OOM error, retrying ({retry_count + 1}/{MAX_RETRIES_ON_OOM})...")
            clear_gpu_memory(verbose=True)
            return infer_single_sample(model, processor, img, retry_count + 1)
        else:
            print(f"  OOM error after {MAX_RETRIES_ON_OOM} retries, skipping sample")
            clear_gpu_memory(verbose=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    except Exception as e:
        print(f"  Error: {e}")
        return {
            "success": False,
            "error": str(e),
        }

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    """Main execution function"""
    print_gpu_memory("Start")
    
    # Load index file
    print(f"\nLoading index from {INDEX_FILE}...")
    with open(INDEX_FILE) as f:
        index_rows = list(csv.DictReader(f))
    
    # Apply smoke test sampling
    if SMOKE_TEST_N is not None and len(index_rows) > SMOKE_TEST_N:
        rng = np.random.default_rng(42)
        sampled_idx = rng.choice(len(index_rows), size=SMOKE_TEST_N, replace=False)
        index_rows = [index_rows[i] for i in sampled_idx]
        print(f"SMOKE TEST MODE: running on {len(index_rows)} samples")
    else:
        print(f"Running on all {len(index_rows)} samples")
    
    # Load model
    processor, model = load_model()
    global real_token_id, fake_token_id
    real_token_id, fake_token_id = find_label_token_ids(processor)
    
    print_gpu_memory("After model load")
    
    # Inference loop
    rows = []
    failed_samples = []
    
    for idx, r in enumerate(tqdm(index_rows, desc="Processing")):
        path = r["path"]
        
        # Load image
        try:
            img = sample_frame_pil(path)
        except Exception as e:
            print(f"\nError loading image {path}: {e}")
            rows.append({
                "path": path,
                "y_qwen": FAKE_LABEL,
                "y_qwen_text": UNK_LABEL,
                "p_fake_qwen": None,
                "p_real_qwen": None,
                "qwen_margin": None,
                "qwen_input_tokens": None,
                "qwen_output_tokens": None,
                "latency_qwen_ms": None,
                "raw": f"LOAD_ERROR: {str(e)}",
            })
            continue
        
        # Run inference
        result = infer_single_sample(model, processor, img)
        
        if result["success"]:
            rows.append({
                "path": path,
                "y_qwen": result["y_score"],
                "y_qwen_text": result["y_text"],
                "p_fake_qwen": result["p_fake"],
                "p_real_qwen": result["p_real"],
                "qwen_margin": result["margin"],
                "qwen_input_tokens": result["input_tokens"],
                "qwen_output_tokens": result["output_tokens"],
                "latency_qwen_ms": result["latency_ms"],
                "raw": result["raw_output"],
            })
        else:
            failed_samples.append(path)
            rows.append({
                "path": path,
                "y_qwen": FAKE_LABEL,
                "y_qwen_text": UNK_LABEL,
                "p_fake_qwen": None,
                "p_real_qwen": None,
                "qwen_margin": None,
                "qwen_input_tokens": None,
                "qwen_output_tokens": None,
                "latency_qwen_ms": None,
                "raw": f"INFERENCE_ERROR: {result.get('error', 'Unknown')}",
            })
        
        # Periodic memory cleanup
        if (idx + 1) % CLEAR_CACHE_EVERY_N == 0:
            clear_gpu_memory(verbose=False)
        
        # Debug: print first 5 results
        if idx < 5:
            print(f"\n[Sample {idx}] Path: {os.path.basename(path)}")
            print(f"  Output: {result.get('raw_output', 'N/A')[:100]}")
            if result.get("p_fake") is not None:
                print(f"  p_fake={result['p_fake']:.3f}, p_real={result['p_real']:.3f}")
            print(f"  Prediction: {'FAKE' if result.get('y_score') == FAKE_LABEL else 'REAL' if result.get('y_score') == REAL_LABEL else 'UNK'}")
    
    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Output file: {OUT_FILE}")
    print(f"Total samples: {len(df)}")
    print(f"Failed samples: {len(failed_samples)}")
    
    if len(df) > 0:
        # Score-based predictions
        score_fake_count = (df['y_qwen'] == FAKE_LABEL).sum()
        score_real_count = (df['y_qwen'] == REAL_LABEL).sum()
        print(f"\nScore-based predictions:")
        print(f"  REAL: {score_real_count} ({score_real_count/len(df)*100:.1f}%)")
        print(f"  FAKE: {score_fake_count} ({score_fake_count/len(df)*100:.1f}%)")
        
        # Text-based predictions
        text_fake_count = (df['y_qwen_text'] == FAKE_LABEL).sum()
        text_real_count = (df['y_qwen_text'] == REAL_LABEL).sum()
        text_unk_count = (df['y_qwen_text'] == UNK_LABEL).sum()
        print(f"\nText-based predictions:")
        print(f"  REAL: {text_real_count} ({text_real_count/len(df)*100:.1f}%)")
        print(f"  FAKE: {text_fake_count} ({text_fake_count/len(df)*100:.1f}%)")
        print(f"  UNK:  {text_unk_count} ({text_unk_count/len(df)*100:.1f}%)")
        
        # Probabilities
        if df['p_fake_qwen'].notna().any():
            print(f"\nScore statistics:")
            print(f"  Mean p_fake: {df['p_fake_qwen'].dropna().mean():.3f}")
            print(f"  Mean margin: {df['qwen_margin'].dropna().mean():.3f}")
        
        # Token statistics
        if df['qwen_input_tokens'].notna().any():
            print(f"\nToken statistics:")
            print(f"  Mean input tokens: {df['qwen_input_tokens'].dropna().mean():.1f}")
            print(f"  Mean output tokens: {df['qwen_output_tokens'].dropna().mean():.1f}")
        
        # Latency
        if df['latency_qwen_ms'].notna().any():
            print(f"\nPerformance:")
            print(f"  Mean latency: {df['latency_qwen_ms'].dropna().mean():.1f} ms")
    
    # Ground truth evaluation (if available)
    if "label" in index_rows[0]:
        gt = pd.DataFrame(index_rows)[["path", "label"]]
        gt["label"] = gt["label"].astype(int)
        merged = df.merge(gt, on="path")
        
        fake_mask = merged["label"] == 1
        if fake_mask.sum() > 0:
            fake_recall_score = (merged.loc[fake_mask, "y_qwen"] == 1).mean()
            print(f"\nEvaluation on {len(merged)} samples with ground truth:")
            print(f"  Fake recall (score): {fake_recall_score:.3f} ({fake_mask.sum()} fakes)")
        
        real_mask = merged["label"] == 0
        if real_mask.sum() > 0:
            real_recall_score = (merged.loc[real_mask, "y_qwen"] == 0).mean()
            print(f"  Real recall (score): {real_recall_score:.3f} ({real_mask.sum()} reals)")
    
    print(f"{'='*60}")
    
    # Cleanup
    cleanup_model(model, processor)
    print_gpu_memory("End")

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        clear_gpu_memory(verbose=True)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory(verbose=True)