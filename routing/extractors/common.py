"""
Shared utilities for feature extractors and experiment scripts.
"""
import csv
import os
import gc
import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image


DATASETS = {
    "GenBuster": {
        "index": "/workspace/benchmarks_ai_research/benchmark_index_with_gen.csv",
        "is_video": True,
        # Use qwen_preds.csv (2000 rows, full dataset) over qwen_preds_v2.csv (partial)
        "cnnspot_csv": "/workspace/benchmarks_ai_research/routing/bandit_dataset.csv",
        "clip_preds": "/workspace/benchmarks_ai_research/routing/clip_preds.csv",
        "qwen_preds": "/workspace/benchmarks_ai_research/routing/qwen_preds.csv",
        "val_paths": "/workspace/benchmarks_ai_research/routing/bandit_val_paths.npy",
        "prefix": "",
    },
    "SD14": {
        "index": "/workspace/benchmarks_ai_research/routing/sd14_index.csv",
        "is_video": False,
        "cnnspot_csv": "/workspace/benchmarks_ai_research/routing/sd14_bandit_dataset.csv",
        "clip_preds": "/workspace/benchmarks_ai_research/routing/sd14_clip_preds.csv",
        "qwen_preds": "/workspace/benchmarks_ai_research/routing/sd14_qwen_preds.csv",
        "val_paths": "/workspace/benchmarks_ai_research/routing/sd14_bandit_val_paths.npy",
        "prefix": "sd14_",
    },
    "BigGAN": {
        "index": "/workspace/benchmarks_ai_research/routing/biggan_index.csv",
        "is_video": False,
        "cnnspot_csv": "/workspace/benchmarks_ai_research/routing/biggan_bandit_dataset.csv",
        "clip_preds": "/workspace/benchmarks_ai_research/routing/biggan_clip_preds.csv",
        "qwen_preds": "/workspace/benchmarks_ai_research/routing/biggan_qwen_preds.csv",
        "val_paths": "/workspace/benchmarks_ai_research/routing/biggan_bandit_val_paths.npy",
        "prefix": "biggan_",
    },
}

ROUTING_BASE = "/workspace/benchmarks_ai_research/routing"
RESULTS_DIR = "/workspace/benchmarks_ai_research/routing/results"

LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_CNN_MS + LATENCY_CLIP_MS  # 56.2ms


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_val_paths(cfg, df_paths=None):
    """
    Load val paths from .npy file. If it doesn't exist, fall back to a
    reproducible 30% split of df_paths (path strings).
    Returns a set of path strings.
    """
    npy = cfg.get("val_paths", "")
    if npy and os.path.exists(npy):
        return set(np.load(npy, allow_pickle=True).tolist())
    if df_paths is not None:
        # Deterministic 70/30 split — use same seed so all detectors agree
        rng = np.random.default_rng(42)
        arr = np.array(df_paths)
        idx = rng.choice(len(arr), size=int(len(arr) * 0.30), replace=False)
        return set(arr[idx].tolist())
    return set()


def load_index(index_file):
    with open(index_file) as f:
        return list(csv.DictReader(f))


def sample_frame_pil(path, is_video=True):
    if is_video:
        vr = VideoReader(path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return Image.open(path).convert("RGB")


def apply_jpeg_compression(img, quality):
    """Apply JPEG compression to a PIL Image and return compressed PIL Image."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def clear_gpu():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def bootstrap_ci(correct, n=1000, ci=95):
    from sklearn.utils import resample
    scores = [np.mean(resample(correct)) for _ in range(n)]
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return lo, hi


def evaluate_cascade(df, get_action):
    """
    get_action(row) -> (pred: int, invoke_vlm: bool)
    Returns dict of evaluation metrics.
    """
    correct, latencies, in_tokens, out_tokens = [], [], [], []
    for _, row in df.iterrows():
        pred, invoke_vlm = get_action(row)
        if invoke_vlm:
            lat = LATENCY_CHEAP_MS + row.get("latency_qwen_ms", 0)
            itok = row.get("qwen_input_tokens", 0) or 0
            otok = row.get("qwen_output_tokens", 0) or 0
        else:
            lat = LATENCY_CHEAP_MS
            itok = otok = 0
        correct.append(int(pred == row["y_true"]))
        latencies.append(lat)
        in_tokens.append(itok)
        out_tokens.append(otok)
    lo, hi = bootstrap_ci(correct)
    return {
        "accuracy": np.mean(correct),
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_latency_ms": np.mean(latencies),
        "escalation_rate": np.mean([l > LATENCY_CHEAP_MS for l in latencies]),
        "mean_input_tokens": np.mean(in_tokens),
        "mean_output_tokens": np.mean(out_tokens),
    }


def balanced_accuracy(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


def auroc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def calibration_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = np.mean(y_true[mask] == (y_prob[mask] >= 0.5).astype(int))
        conf = np.mean(y_prob[mask])
        ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)
