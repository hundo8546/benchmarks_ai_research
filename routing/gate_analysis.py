"""
Reviewer 1 Comment 4 — validation gate sensitivity sweep (4b) and three-way
f1/f2/f3 selection extension (4c).

Reuses the per-cell disagreement-subset arrays built by baseline_comparison.py
(same 32 configs: verifier_comparison, e8_results, cross_family_deployment,
limited_data). See claude_context_notes/resubmissionplan.md Sections 3.5/3.6.

Usage:
    python gate_analysis.py
"""
import os

import numpy as np
import pandas as pd

import baseline_comparison as bc
from extractors.common import RESULTS_DIR, ensure_results_dir

N_GRID = [10, 20, 50, 100, 200]
N_RESAMPLES = 1000
SEED = 42


def oracle_decisions(cell):
    y_true, y1, y2, y3 = cell["y_true"], cell["y_cheap1"], cell["y_cheap2"], cell["y_verifier"]
    acc1 = float(np.mean(y1 == y_true))
    acc2 = float(np.mean(y2 == y_true))
    acc3 = float(np.mean(y3 == y_true))
    best_cheap_acc = max(acc1, acc2)

    binary_escalate = acc3 > best_cheap_acc
    three_way_choice = int(np.argmax([acc1, acc2, acc3]))  # 0=cheap1,1=cheap2,2=verifier
    return acc1, acc2, acc3, binary_escalate, three_way_choice


def sweep_cell(cell, rng):
    y_true, y1, y2, y3 = cell["y_true"], cell["y_cheap1"], cell["y_cheap2"], cell["y_verifier"]
    n_dis = len(y_true)
    acc1, acc2, acc3, oracle_binary, oracle_three_way = oracle_decisions(cell)

    rows = []
    for N in N_GRID + ["oracle"]:
        n_sample = n_dis if N == "oracle" else min(N, n_dis)
        binary_matches = 0
        three_way_matches = 0
        for _ in range(N_RESAMPLES):
            if N == "oracle":
                idx = np.arange(n_dis)
            else:
                idx = rng.choice(n_dis, size=n_sample, replace=True)

            a1 = float(np.mean(y1[idx] == y_true[idx]))
            a2 = float(np.mean(y2[idx] == y_true[idx]))
            a3 = float(np.mean(y3[idx] == y_true[idx]))

            calib_binary = a3 > max(a1, a2)
            calib_three_way = int(np.argmax([a1, a2, a3]))

            binary_matches += int(calib_binary == oracle_binary)
            three_way_matches += int(calib_three_way == oracle_three_way)

            if N == "oracle":
                break  # deterministic, no need to resample

        n_trials = 1 if N == "oracle" else N_RESAMPLES
        rows.append({
            "table": cell["table"], "config": cell["config"], "verifier": cell["verifier"],
            "n_disagree": n_dis, "calib_N": N if N != "oracle" else n_dis,
            "binary_agreement_rate": binary_matches / n_trials,
            "three_way_agreement_rate": three_way_matches / n_trials,
            "oracle_binary_escalate": oracle_binary,
            "oracle_three_way_choice": ["cheap1", "cheap2", "verifier"][oracle_three_way],
        })
    return rows


def main():
    ensure_results_dir()
    print("Rebuilding subset cells from baseline_comparison...")
    bc.run_verifier_comparison()
    bc.run_e8_results()
    bc.run_cross_family_deployment()
    bc.run_limited_data()
    bc.run_biggan_shift()

    rng = np.random.default_rng(SEED)
    all_rows = []
    for cell in bc.SUBSETS:
        all_rows.extend(sweep_cell(cell, rng))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "gate_sensitivity_sweep.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows, {len(bc.SUBSETS)} configs x {len(N_GRID)+1} N-values)\n")

    # Summary: at each N, how many configs reach >=90% decision-agreement with oracle
    print("=== Binary gate sensitivity (agreement with oracle escalate/fallback decision) ===")
    summary = df.groupby("calib_N").agg(
        n_configs=("config", "count"),
        mean_agreement=("binary_agreement_rate", "mean"),
        n_configs_ge90pct=("binary_agreement_rate", lambda s: int((s >= 0.90).sum())),
    ).reset_index()
    print(summary.to_string(index=False))

    print("\n=== Three-way gate sensitivity (agreement with oracle f1/f2/f3 selection) ===")
    summary3 = df.groupby("calib_N").agg(
        n_configs=("config", "count"),
        mean_agreement=("three_way_agreement_rate", "mean"),
        n_configs_ge90pct=("three_way_agreement_rate", lambda s: int((s >= 0.90).sum())),
    ).reset_index()
    print(summary3.to_string(index=False))

    n_total_configs = len(bc.SUBSETS)
    for N in N_GRID:
        sub = df[df["calib_N"] == N]
        n_ge90 = int((sub["binary_agreement_rate"] >= 0.90).sum())
        print(f"N={N:>3}: binary gate reaches >=90% oracle-agreement in {n_ge90}/{n_total_configs} configs")

    print("\n=== Oracle three-way choice distribution (how often is each detector 'best' on the disagreement subset) ===")
    oracle_rows = df[df["calib_N"] == df["n_disagree"]][["table", "config", "verifier", "oracle_binary_escalate", "oracle_three_way_choice"]]
    print(oracle_rows.to_string(index=False))
    print("\n" + oracle_rows["oracle_three_way_choice"].value_counts().to_string())


if __name__ == "__main__":
    main()
