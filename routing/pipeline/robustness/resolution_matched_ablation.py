"""
routing/pipeline/robustness/resolution_matched_ablation.py

Resolution-matched FFT ablation (reviewer concern, peerreviewdecision2.txt
point 9): BigGAN fakes are ALL natively 128x128 and SD1.4 fakes are ALL
natively 512x512, while real (ImageNet) images span 100+ native resolutions
(see native_resolution.py's audit). extract_fft.py resizes everything to a
fixed 256x256 before computing FFT/DCT features, so a clean, uniform
upsample/downsample factor could itself be a detectable, non-forensic signal
that inflates FFT's apparent accuracy.

This ablation removes that confound on the REAL side only, WITHOUT touching
the already-trained FFT probe (no retraining):
  - fakes: used exactly as in the original pipeline (native 128x128 / 512x512
    -> resize to 256x256 -> extract_fft.extract_features).
  - reals (resolution-matched condition only): native resolution -> resized
    DOWN/UP to the dataset's fake native resolution (128x128 for BigGAN,
    512x512 for SD1.4) -> THEN the same 256x256 resize + extract_fft feature
    pipeline as everything else. This gives reals the same
    "resample-to-a-clean-fixed-size" signature that fakes have, while leaving
    pixel CONTENT (the real photograph) untouched.

The SAME canonical eval split and the SAME existing {prefix}fft_probe.pkl
(clf, scaler) are used for both conditions -- only the real-image
preprocessing differs. If accuracy collapses under resolution-matching, the
original number partly reflects the resolution confound. If accuracy stays
near-ceiling, the FFT signal is robust to it.

Output: routing/results/resolution_matched_ablation.csv
  columns: dataset, condition [original / resolution-matched], fft_accuracy, n
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extractors"))
from extractors.common import DATASETS, RESULTS_DIR, ROUTING_BASE, load_index, ensure_results_dir
from extractors.extract_fft import extract_features as fft_extract_features
from pipeline.splits import get_canonical_eval_paths

# Native resolution that this dataset's generator produces fakes at --
# confirmed empirically by native_resolution.py (100% of sampled fakes).
FAKE_NATIVE_RES = {
    "BigGAN": 128,
    "SD14": 512,
}


def load_fft_probe(ds_name):
    cfg = DATASETS[ds_name]
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_probe.pkl")
    if not os.path.exists(probe_pkl):
        raise FileNotFoundError(f"Missing existing FFT probe: {probe_pkl}")
    return joblib.load(probe_pkl)  # (clf, scaler)


def run_dataset(ds_name):
    target_res = FAKE_NATIVE_RES[ds_name]
    cfg = DATASETS[ds_name]
    clf, scaler = load_fft_probe(ds_name)

    eval_paths = get_canonical_eval_paths(ds_name)
    index_rows = load_index(cfg["index"])
    eval_rows = [r for r in index_rows if r["path"] in eval_paths]
    print(f"{ds_name}: n_eval={len(eval_rows)} (target native res = {target_res}x{target_res})")

    feats_orig, feats_matched, labels = [], [], []
    for r in tqdm(eval_rows, desc=f"{ds_name} FFT resolution-matched ablation"):
        path = r["path"]
        label = int(r["label"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Skip {path}: {e}")
            continue

        # Original condition: exactly the standard extractor pipeline.
        f_orig = fft_extract_features(img)

        # Resolution-matched condition: only reals get an extra
        # resize-to-target-native-res step before the standard pipeline.
        # Fakes are already natively at target_res, so they are unchanged
        # (resizing 128->128 or 512->512 is a no-op up to interpolation,
        # but we apply it uniformly for a clean comparison).
        img_matched = img.resize((target_res, target_res), Image.BICUBIC)
        f_matched = fft_extract_features(img_matched)

        feats_orig.append(f_orig)
        feats_matched.append(f_matched)
        labels.append(label)

    X_orig = np.stack(feats_orig).astype(np.float32)
    X_matched = np.stack(feats_matched).astype(np.float32)
    y = np.array(labels)

    y_pred_orig = clf.predict(scaler.transform(X_orig))
    y_pred_matched = clf.predict(scaler.transform(X_matched))

    acc_orig = float(np.mean(y_pred_orig == y))
    acc_matched = float(np.mean(y_pred_matched == y))

    print(f"  original accuracy          = {acc_orig:.4f}")
    print(f"  resolution-matched accuracy = {acc_matched:.4f}  (reals only resized "
          f"to {target_res}x{target_res} before the standard 256x256 FFT pipeline; "
          f"fakes unchanged)")

    return [
        {"dataset": ds_name, "condition": "original", "fft_accuracy": acc_orig, "n": len(y)},
        {"dataset": ds_name, "condition": "resolution-matched", "fft_accuracy": acc_matched, "n": len(y)},
    ]


def main():
    ensure_results_dir()
    rows = []
    for ds in ["BigGAN", "SD14"]:
        rows.extend(run_dataset(ds))
    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "resolution_matched_ablation.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
