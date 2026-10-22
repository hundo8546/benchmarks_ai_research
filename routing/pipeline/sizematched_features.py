"""
routing/pipeline/sizematched_features.py

Step 4.1 of the SDXL/FLUX resolution-confound control (remainingupdates1.md):
extract FFT, SigLIP2 and DINOv2 features for the size-matched SDXL/FLUX images
(pipeline/robustness/build_sizematched_sdxl_flux.py) with the EXISTING E8
extractors, experiments/e8_modern_diffusion.extract_features, unchanged. The
only difference from the original E8 configs is the index (size-matched
copies) and the output prefix:

  routing/{sdxl,flux}_sizematched_{fft,siglip2,dinov2,clip}_features.csv

(e8.extract_features also writes CLIP features; they are not used by the
size-matched tables.) Feature rows keep the size-matched path in "path";
results/sizematched_image_map.csv maps it back to the original image.

GPU. Usage: python pipeline/sizematched_features.py [--datasets SDXL FLUX]
"""
import argparse
import os
import sys

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROUTING_BASE)
sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))


def sizematched_config(e8, ds_name):
    cfg = dict(e8.E8_DATASETS[ds_name])
    cfg["index"] = os.path.join(ROUTING_BASE, f"sizematched_{ds_name.lower()}_index.csv")
    cfg["prefix"] = f"{ds_name.lower()}_sizematched_"
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["SDXL", "FLUX"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import e8_modern_diffusion as e8
    for ds_name in a.datasets:
        print(f"=== {ds_name} (size-matched) ===")
        e8.extract_features(ds_name, sizematched_config(e8, ds_name), device=a.device)
