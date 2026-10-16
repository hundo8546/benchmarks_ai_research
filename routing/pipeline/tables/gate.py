"""
routing.pipeline.tables.gate — tab_sensitivity, tab_threeway.

Ports routing/gate_analysis.py (validation-gate sensitivity sweep + the
three-way f1/f2/f3 selection extension) into importable form. Both tables
come from the same underlying sweep (results/gate_sensitivity_sweep.csv);
this module splits it into the two column views the paper actually tables:

  tab_sensitivity.csv -- binary gate: does a calibration set of size N agree
                          with the oracle escalate/fallback decision?
  tab_threeway.csv    -- three-way gate: does a calibration set of size N
                          agree with the oracle f1/f2/f3 (cheap1/cheap2/
                          verifier) selection?

CPU-only: gate_analysis.py's own sweep logic has no torch/cv2/decord
dependency itself, but it reuses baseline_comparison.py's per-cell SUBSETS
(see pipeline.tables.baselines), which does pull in those heavy deps via
extractors.common. That import is deferred to inside compute_sweep_raw(),
not at this module's top level.
"""
import os
import sys

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_GRID = [10, 20, 50, 100, 200]
N_RESAMPLES = 1000
SEED = 42


def compute_sweep_raw():
    """Rebuild the full gate_sensitivity_sweep rows (one row per config x
    verifier x calib_N, including the 'oracle' row at calib_N == n_disagree)
    by reusing gate_analysis.py's oracle_decisions()/sweep_cell() directly
    against pipeline.tables.baselines' SUBSETS."""
    sys.path.insert(0, ROUTING_BASE)
    from pipeline.tables.baselines import compute_baselines_raw
    _, bc = compute_baselines_raw()

    sys.path.insert(0, os.path.join(ROUTING_BASE))
    import gate_analysis as ga

    rng = np.random.default_rng(SEED)
    all_rows = []
    for cell in bc.SUBSETS:
        all_rows.extend(ga.sweep_cell(cell, rng))
    return pd.DataFrame(all_rows)


def compute_sensitivity():
    """tab_sensitivity: binary escalate/fallback gate agreement-with-oracle
    by calibration sample size N."""
    df = compute_sweep_raw()
    cols = ["table", "config", "verifier", "n_disagree", "calib_N",
            "binary_agreement_rate", "oracle_binary_escalate"]
    return df[cols].reset_index(drop=True)


def compute_threeway():
    """tab_threeway: three-way f1/f2/f3 selection gate agreement-with-oracle
    by calibration sample size N."""
    df = compute_sweep_raw()
    cols = ["table", "config", "verifier", "n_disagree", "calib_N",
            "three_way_agreement_rate", "oracle_three_way_choice"]
    return df[cols].reset_index(drop=True)
