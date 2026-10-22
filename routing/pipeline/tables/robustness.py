"""
routing.pipeline.tables.robustness — tab_robustness.

FFT / SigLIP2 / DINOv2 accuracy under JPEG compression, Gaussian blur and a
down/up-scale round trip, on the canonical in-domain eval sets. Reads the outputs of
pipeline/robustness/e7_compression_extended.py and e9_postprocessing_robustness.py.
"""
import os

import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROUTING_BASE, "results")

DETECTORS = {"FFT": "FFT", "SIGLIP2": "SigLIP2", "DINOV2": "DINOv2"}
COLUMNS = [
    ("clean", "e7", "raw"), ("jpeg_q95", "e7", "jpeg95"), ("jpeg_q75", "e7", "jpeg75"),
    ("jpeg_q50", "e7", "jpeg50"), ("blur_sigma1", "e9", "sigma_1.0"), ("blur_sigma2", "e9", "sigma_2.0"),
    ("rescale_x0.5", "e9", "factor_0.5"), ("rescale_x0.25", "e9", "factor_0.25"),
]


def compute_robustness():
    e7 = pd.read_csv(os.path.join(RESULTS, "e7_compression_extended.csv"))
    e9 = pd.read_csv(os.path.join(RESULTS, "e9_postprocessing_robustness.csv"))
    e7 = e7.rename(columns={"compression": "condition"})
    e9 = e9.rename(columns={"severity_level": "condition"})
    src = {"e7": e7, "e9": e9}
    rows = []
    for ds in ["GenBuster", "SD14", "BigGAN"]:
        for det, label in DETECTORS.items():
            row = {"dataset": ds, "detector": label}
            for col, which, cond in COLUMNS:
                d = src[which]
                hit = d[(d["dataset"] == ds) & (d["detector"] == det) & (d["condition"] == cond)]
                if len(hit) != 1:
                    raise ValueError(f"{ds}/{label}/{cond}: expected 1 row in {which}, found {len(hit)}")
                row[col] = float(hit["accuracy"].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)
