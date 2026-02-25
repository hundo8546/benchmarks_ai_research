import torch
import numpy as np
import csv
import cv2
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from decord import VideoReader, cpu

# -------------------------
# CONFIG
# -------------------------

DEVICE = "cuda"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "You are a forensic AI detector. "
    "Is this content REAL or FAKE? "
    "Answer with one word: REAL or FAKE."
)

torch.cuda.empty_cache()

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer
print("Model loaded.\n")

# -------------------------
# FRAME SAMPLING
# -------------------------

def sample_frames(path, n=16):
    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, n).astype(int)
    frames = vr.get_batch(idx).asnumpy()
    return np.array([cv2.resize(f, (448, 448)) for f in frames])

def sample_single_frame(path):
    vr = VideoReader(path, ctx=cpu(0))
    frame = vr[0].asnumpy()
    return cv2.resize(frame, (448, 448))

# -------------------------
# LOGIT CLASSIFICATION
# -------------------------

def classify(inputs):
    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    next_token_logits = logits[:, -1, :]

    real_id = tokenizer.encode("real", add_special_tokens=False)[0]
    fake_id = tokenizer.encode("fake", add_special_tokens=False)[0]

    real_score = next_token_logits[0, real_id].item()
    fake_score = next_token_logits[0, fake_id].item()

    return 0 if real_score > fake_score else 1

# -------------------------
# MODES
# -------------------------

def predict_image(path):
    frame = sample_single_frame(path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[frame],
        return_tensors="pt"
    ).to(DEVICE)

    return classify(inputs)

def predict_video(path):
    frames = sample_frames(path, 16)

    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": PROMPT}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        videos=[frames],
        return_tensors="pt"
    ).to(DEVICE)

    return classify(inputs)

def predict_fusion(path):
    frame = sample_single_frame(path)
    frames = sample_frames(path, 16)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "video"},
            {"type": "text", "text": PROMPT}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[frame],
        videos=[frames],
        return_tensors="pt"
    ).to(DEVICE)

    return classify(inputs)

# -------------------------
# EVALUATION
# -------------------------

results = {
    "image_real": [],
    "image_fake": [],
    "video_real": [],
    "video_fake": [],
    "fusion_real": [],
    "fusion_fake": []
}

with open("benchmark_index_with_gen.csv") as f:
    rows = list(csv.DictReader(f))

print("Starting full Table 2 evaluation...\n")

for i, row in enumerate(tqdm(rows)):
    path = row["path"]
    label = int(row["label"])  # 0 real, 1 fake

    img_pred = predict_image(path)
    vid_pred = predict_video(path)
    fus_pred = predict_fusion(path)

    # Debug print first 10 samples
    if i < 10:
        print(f"\nSample {i}")
        print("GT:", label)
        print("Image Pred:", img_pred)
        print("Video Pred:", vid_pred)
        print("Fusion Pred:", fus_pred)

    if label == 0:
        results["image_real"].append(img_pred == label)
        results["video_real"].append(vid_pred == label)
        results["fusion_real"].append(fus_pred == label)
    else:
        results["image_fake"].append(img_pred == label)
        results["video_fake"].append(vid_pred == label)
        results["fusion_fake"].append(fus_pred == label)

# -------------------------
# METRICS
# -------------------------

def mean(x): return np.mean(x)

image_real = mean(results["image_real"])
image_fake = mean(results["image_fake"])
video_real = mean(results["video_real"])
video_fake = mean(results["video_fake"])
fusion_real = mean(results["fusion_real"])
fusion_fake = mean(results["fusion_fake"])

print("\n===== TABLE 2 RESULTS =====")
print("\nImage Only:")
print("  Real:", round(image_real, 4))
print("  Fake:", round(image_fake, 4))
print("  Overall:", round(np.mean([image_real, image_fake]), 4))

print("\nVideo Only:")
print("  Real:", round(video_real, 4))
print("  Fake:", round(video_fake, 4))
print("  Overall:", round(np.mean([video_real, video_fake]), 4))

print("\nImage + Video:")
print("  Real:", round(fusion_real, 4))
print("  Fake:", round(fusion_fake, 4))
print("  Overall:", round(np.mean([fusion_real, fusion_fake]), 4))