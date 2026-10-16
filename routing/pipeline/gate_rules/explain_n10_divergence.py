"""
Standalone, read-only diagnostic for the Reviewer 1 point-3 question:
why do the binary validation gate and the three-way f1/f2/f3 selector
report different oracle-agreement counts at N=10 (13/35 vs 12/35) but
identical counts at N=20/50/100/200?

This script does NOT modify gate_analysis.py or baseline_comparison.py.
It imports them (gate_analysis.oracle_decisions, baseline_comparison.SUBSETS)
and re-runs the EXACT SAME bootstrap loop, in the exact same cell/N
iteration order, with the exact same seeded RNG (np.random.default_rng(42)),
so that the per-trial random draws are byte-for-byte identical to the ones
that produced routing/results/gate_sensitivity_sweep.csv. It additionally
records, for every trial, which rule's binary call matched/mismatched
the oracle, so we can show the mechanism rather than just the rates.

Usage:
    python explain_n10_divergence.py
"""
import os
import sys

import numpy as np

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROUTING_BASE)

import baseline_comparison as bc  # noqa: E402
import gate_analysis as ga  # noqa: E402

N_GRID = ga.N_GRID
N_RESAMPLES = ga.N_RESAMPLES
SEED = ga.SEED

TARGET = dict(table="limited_data", config="SD14/SigLIP2/frac=100%", verifier="Qwen2.5-VL-7B")


def sweep_cell_instrumented(cell, rng, capture_N=None):
    """Mirrors gate_analysis.sweep_cell exactly (same RNG draw order), but
    optionally captures every per-trial (a1, a2, a3, calib_binary,
    calib_three_way) tuple at one N value for a target cell."""
    y_true, y1, y2, y3 = cell["y_true"], cell["y_cheap1"], cell["y_cheap2"], cell["y_verifier"]
    n_dis = len(y_true)
    acc1, acc2, acc3, oracle_binary, oracle_three_way = ga.oracle_decisions(cell)

    is_target = (cell["table"] == TARGET["table"] and cell["config"] == TARGET["config"]
                 and cell["verifier"] == TARGET["verifier"])

    captured = []
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

            bin_match = int(calib_binary == oracle_binary)
            three_match = int(calib_three_way == oracle_three_way)
            binary_matches += bin_match
            three_way_matches += three_match

            if is_target and capture_N is not None and N == capture_N:
                captured.append((a1, a2, a3, calib_binary, calib_three_way, bin_match, three_match))

            if N == "oracle":
                break

        n_trials = 1 if N == "oracle" else N_RESAMPLES
        rows.append({
            "table": cell["table"], "config": cell["config"], "verifier": cell["verifier"],
            "n_disagree": n_dis, "calib_N": N if N != "oracle" else n_dis,
            "binary_agreement_rate": binary_matches / n_trials,
            "three_way_agreement_rate": three_way_matches / n_trials,
        })
    return rows, captured, (acc1, acc2, acc3, oracle_binary, oracle_three_way)


def main():
    print("Rebuilding subset cells from baseline_comparison (same as gate_analysis.main())...")
    bc.run_verifier_comparison()
    bc.run_e8_results()
    bc.run_cross_family_deployment()
    bc.run_limited_data()
    bc.run_biggan_shift()

    rng = np.random.default_rng(SEED)
    target_capture = None
    target_oracle = None
    all_flip_configs = []

    all_rows = []
    for cell in bc.SUBSETS:
        rows, captured, oracle_vals = sweep_cell_instrumented(cell, rng, capture_N=10)
        all_rows.extend(rows)
        if captured:
            target_capture = captured
            target_oracle = oracle_vals

    import pandas as pd
    df = pd.DataFrame(all_rows)
    sub10 = df[df["calib_N"] == 10].copy()
    sub10["bin_ge90"] = sub10["binary_agreement_rate"] >= 0.90
    sub10["three_ge90"] = sub10["three_way_agreement_rate"] >= 0.90
    n_bin = int(sub10["bin_ge90"].sum())
    n_three = int(sub10["three_ge90"].sum())
    print(f"\nReproduced N=10 counts: binary {n_bin}/35, three-way {n_three}/35 "
          f"(paper reports 13/35, 12/35)")

    flips = sub10[sub10["bin_ge90"] != sub10["three_ge90"]]
    print("\nConfig(s) where the >=90% threshold call differs between the two rules at N=10:")
    print(flips[["table", "config", "verifier", "binary_agreement_rate", "three_way_agreement_rate"]]
          .to_string(index=False))

    acc1, acc2, acc3, oracle_binary, oracle_three = target_oracle
    print(f"\n--- Full-subset oracle for target cell {TARGET} ---")
    print(f"n_disagree={len(bc.SUBSETS[0]['y_true']) if False else '(see below)'}")
    print(f"acc1 (cheap1) = {acc1:.4f}")
    print(f"acc2 (cheap2) = {acc2:.4f}")
    print(f"acc3 (verifier) = {acc3:.4f}")
    print(f"oracle_binary_escalate = {oracle_binary}  (acc3 > max(acc1,acc2)? {acc3:.4f} > {max(acc1,acc2):.4f})")
    print(f"oracle_three_way_choice = {['cheap1','cheap2','verifier'][oracle_three]}")

    cap = np.array(target_capture, dtype=object)
    a1s = np.array([c[0] for c in target_capture])
    a2s = np.array([c[1] for c in target_capture])
    a3s = np.array([c[2] for c in target_capture])
    calib_binaries = np.array([c[3] for c in target_capture])
    calib_three = np.array([c[4] for c in target_capture])
    bin_matches = np.array([c[5] for c in target_capture])
    three_matches = np.array([c[6] for c in target_capture])

    print(f"\n--- N=10 bootstrap trial breakdown for target cell (1000 resamples) ---")
    print(f"binary_agreement_rate  = {bin_matches.mean():.3f}  ({bin_matches.sum()} / 1000 match oracle_binary_escalate={oracle_binary})")
    print(f"three_way_agreement_rate = {three_matches.mean():.3f}  ({three_matches.sum()} / 1000 match oracle_three_way_choice=cheap2)")

    # Decompose the three-way mismatches: how many occur WHILE binary still matches?
    binary_matches_mask = bin_matches.astype(bool)
    three_mismatch_mask = ~three_matches.astype(bool)
    extra_three_mismatches = binary_matches_mask & three_mismatch_mask
    print(f"\nTrials where binary rule matches oracle (escalate boolean correct) "
          f"but three-way rule still mismatches oracle (wrong cheap detector identified): "
          f"{extra_three_mismatches.sum()} / 1000")

    # Of those, what did three-way pick instead of cheap2, and why (a1 vs a2 relationship)?
    among_extra = calib_three[extra_three_mismatches]
    vals, counts = np.unique(among_extra, return_counts=True)
    label = {0: "cheap1", 1: "cheap2", 2: "verifier"}
    print("Breakdown of what three-way picked in those trials:",
          {label[v]: int(c) for v, c in zip(vals, counts)})

    # Among all 1000 trials, how often does a1 (cheap1 resampled acc) exceed a2 (cheap2 resampled acc)?
    a1_beats_a2 = (a1s > a2s)
    print(f"\nAcross all 1000 N=10 resamples: a1 > a2 (cheap1 resampled-acc exceeds cheap2) "
          f"in {a1_beats_a2.sum()} / 1000 trials (true accs are acc1={acc1:.3f}, acc2={acc2:.3f})")
    a1_eq_a2 = (a1s == a2s)
    print(f"a1 == a2 (exact tie, argmax breaks toward cheap1/index0) in {a1_eq_a2.sum()} / 1000 trials")

    # And how often does a3 exceed max(a1,a2) -- this is what the binary rule's metric alone depends on
    max12 = np.maximum(a1s, a2s)
    a3_beats_max12 = (a3s > max12)
    print(f"a3 > max(a1,a2)  in {a3_beats_max12.sum()} / 1000 trials "
          f"(oracle says this should be False; these trials drive binary's own {1000-bin_matches.sum()} mismatches)")


if __name__ == "__main__":
    main()
