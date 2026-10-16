"""
routing.pipeline.tables.transfer — tab_transfer.

Ports routing/bandit_transfer_fft_siglip2.py (cross-dataset learned-escalation-
classifier transfer matrix using the fair FFT+SigLIP2+Qwen learned escalation
classifier) into an importable form that returns a DataFrame instead of only
printing + writing
bandit_transfer_fft_siglip2.csv directly.

CPU-only: bandit_transfer_fft_siglip2.py imports
retrain_bandits_fft_siglip2.py, which imports extractors.common (torch/cv2/
decord at module load). Deferred to inside compute_transfer(), not this
module's top level.
"""
import os
import sys

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_NAMES = ["GenBuster", "SD14", "BigGAN"]
SEED = 42


def compute_transfer():
    """tab_transfer: train x test learned-escalation-classifier transfer
    accuracy/escalation-rate matrix (9 rows: 3x3 train/test pairs, GenBuster diagonal caveated as
    internal-split per bandit_transfer_fft_siglip2.py's own docstring)."""
    sys.path.insert(0, ROUTING_BASE)
    import bandit_transfer_fft_siglip2 as bt

    data = {}
    for ds_name, prefix, qwen_filename, val_paths_file, _ in bt.DATASETS:
        df, val_paths = bt.load_dataset(ds_name, qwen_filename, val_paths_file)
        data[ds_name] = (df, val_paths)

    models = {}
    for ds_name in ["SD14", "BigGAN"]:
        df, val_paths = data[ds_name]
        model, scaler = bt.fit_full_bandit(ds_name, df, val_paths)
        models[ds_name] = (model, scaler)

    df_gb, val_paths_gb = data["GenBuster"]
    model_gb, scaler_gb, _, _ = bt.train_bandit(df_gb, val_paths_gb)
    models["GenBuster"] = (model_gb, scaler_gb)

    rows = []
    for train_name in DATASET_NAMES:
        model, scaler = models[train_name]
        for test_name in DATASET_NAMES:
            df_test, val_paths_test = data[test_name]
            acc, esc = bt.eval_bandit(model, scaler, df_test, val_paths_test)
            rows.append({
                "train": train_name, "test": test_name,
                "accuracy": acc, "escalation_rate": esc,
                "in_domain": train_name == test_name,
                "note": ("GenBuster learned escalation classifier fit on internal split of its own "
                          "val set (no train-split Qwen coverage)") if train_name == "GenBuster" else "",
            })
    return pd.DataFrame(rows)
