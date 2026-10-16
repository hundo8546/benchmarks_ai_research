"""
routing.pipeline.tables.baselines — tab_verifier_comparison, tab_baselines,
tab_e8_results, tab_cross_family_deployment, tab_limited_data,
tab_biggan_shift.

Ports routing/baseline_comparison.py (which already writes
results/baseline_comparison_summary.csv covering all five table families as
one long DataFrame with a `table` column) into an importable form, split back
out per table to match the paper's actual table structure:

  tab_baselines.csv               -- the full, unfiltered summary (every
                                      config x verifier cell across all five
                                      table families) -- i.e. "the baseline
                                      comparison table" as a whole.
  tab_verifier_comparison.csv     -- rows where table == "verifier_comparison"
  tab_e8_results.csv              -- rows where table == "e8_results"
  tab_cross_family_deployment.csv -- rows where table == "cross_family_deployment"
  tab_limited_data.csv            -- rows where table == "limited_data"
  tab_biggan_shift.csv            -- rows where table == "biggan_shift"

Why both: tab_baselines.csv is the reviewer-facing "does the verifier clear
random/majority/best-cheap" table as a single artifact (easy to eyeball
overall pass rates), while the per-table CSVs match each table's own LaTeX
label 1:1 so generate_paper_results.py's "one CSV per \\label{}" convention
holds even though all five share one underlying computation.

CPU-only: baseline_comparison.py (and the e8_modern_diffusion.py /
e6_limited_data.py / cross_family_deployment.py it imports) pull in
extractors.common, which imports torch/cv2/decord at module load. That
import is deferred to inside compute_baselines_raw(), not at this module's
top level, for the same reason as pipeline/tables/main_results.py.
"""
import os
import sys

import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TABLE_FAMILIES = [
    "verifier_comparison", "e8_results", "cross_family_deployment",
    "limited_data", "biggan_shift",
]


def compute_baselines_raw():
    """Run baseline_comparison.py's five run_*() builders and return the full
    ROWS DataFrame (same content as results/baseline_comparison_summary.csv).
    Heavy imports (torch/cv2/decord, via extractors.common) happen here, on
    first call, not at module import time."""
    sys.path.insert(0, ROUTING_BASE)
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))
    import baseline_comparison as bc

    # Reset any state from a prior call in the same process (ROWS/SUBSETS are
    # module-level lists that bc's run_*() functions append to).
    bc.ROWS.clear()
    bc.SUBSETS.clear()

    bc.run_verifier_comparison()
    bc.run_e8_results()
    bc.run_cross_family_deployment()
    bc.run_limited_data()
    bc.run_biggan_shift()

    cols = ["table", "config", "verifier", "n", "verifier_acc",
            "random_mean", "random_ci_lo", "random_ci_hi", "clears_random",
            "majority_acc", "verifier_minus_majority_ci_lo", "verifier_minus_majority_ci_hi", "clears_majority",
            "best_cheap_acc", "best_cheap_source",
            "verifier_minus_best_cheap_ci_lo", "verifier_minus_best_cheap_ci_hi", "clears_best_cheap"]
    return pd.DataFrame(bc.ROWS)[cols], bc


def compute_baselines():
    """tab_baselines: the full, unfiltered summary across all five families."""
    df, _ = compute_baselines_raw()
    return df


def compute_verifier_comparison():
    df, _ = compute_baselines_raw()
    return df[df["table"] == "verifier_comparison"].reset_index(drop=True)


def compute_e8_results():
    df, _ = compute_baselines_raw()
    return df[df["table"] == "e8_results"].reset_index(drop=True)


def compute_cross_family_deployment():
    df, _ = compute_baselines_raw()
    return df[df["table"] == "cross_family_deployment"].reset_index(drop=True)


def compute_limited_data():
    df, _ = compute_baselines_raw()
    return df[df["table"] == "limited_data"].reset_index(drop=True)


def compute_biggan_shift():
    df, _ = compute_baselines_raw()
    return df[df["table"] == "biggan_shift"].reset_index(drop=True)
