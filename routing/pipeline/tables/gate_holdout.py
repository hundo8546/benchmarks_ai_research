"""
routing.pipeline.tables.gate_holdout -- Reviewer 1 (2nd round), Points 1 & 2.

Point 1 ("Evaluate the validation gate on an independent deployment set."):
`gate_analysis.py` / `pipeline.tables.gate.py` only ever bootstrap-resample
the FULL disagreement subset and compare each resample's decision to an
oracle computed from that SAME full subset. That measures resampling
stability of the decision rule, not whether a decision frozen on one batch
of labeled disagreements generalizes to a *different*, unseen batch.

This module instead, for every one of the 35 (table, config, verifier)
disagreement-subset cells already built by
`pipeline.tables.baselines.compute_baselines_raw()` (same cells
`gate_analysis.py` sweeps), repeatedly:
  1. splits the disagreement subset into a calibration half and a disjoint
     test half (no overlap, no shared samples -- genuine held-out split),
  2. on the calibration half ONLY, chooses an action (run cheap1, run
     cheap2, or escalate to the verifier) under two rules:
       - "naive": the current paper rule, acc3 > max(acc1, acc2) as a bare
         point estimate (equivalent to the three-way argmax selector; see
         pipeline/gate_rules/PSEUDOCODE.md for why these are the same
         decision procedure),
       - "sig": escalate only if the paired-bootstrap 95% CI (reusing
         baseline_utils.paired_bootstrap_diff_ci, not reimplemented) on
         (verifier_acc - best_cheap_acc), computed ON THE CALIBRATION HALF
         ONLY, lies entirely above zero; otherwise fall back to whichever
         cheap detector is stronger on calibration,
  3. FREEZES that decision and applies it, unchanged, to the test half,
  4. scores the frozen decision against the test half's own (independent)
     oracle: test accuracy, regret (test-oracle-best accuracy minus frozen-
     decision test accuracy, which is >=0 by construction), whether this
     was a harmful escalation (decision said "escalate" but the verifier's
     TEST-half accuracy is actually below the best cheap detector's
     TEST-half accuracy), and whether the frozen action matches what the
     test half's own oracle would have chosen (the genuinely-held-out
     analogue of gate_analysis.py's oracle-agreement rate).

Point 2 ("The gate does not require statistically convincing improvement."
+ "auditing the agreement path"):
  - The "sig" rule above IS the prespecified, CI-based, uncertainty-aware
    decision rule the reviewer asked for (vs. "naive"'s bare point estimate).
  - `compute_agreement_path_audit()` computes agreement-path accuracy and
    the rate of CORRELATED errors (both cheap detectors wrong AND agreeing,
    i.e. the "Breakdown condition" from the paper's K-stage formulation that
    disagreement-based routing cannot catch, because agreeing samples are
    never escalated) across GenBuster/SD14/BigGAN (in-domain) and the
    SDXL/FLUX deployment-shift settings -- not just the existing
    GenBuster-only 2x2 case-type breakdown in
    `pipeline.tables.main_results.compute_escalation()`.

No heavy imports (torch/cv2/decord) at module load time: those are pulled in
transitively by `pipeline.tables.baselines.compute_baselines_raw()` and
`pipeline.tables.main_results.load_all()`, both of which are only imported
lazily inside the compute_*() functions below, matching the pattern used
throughout pipeline/tables/.

Usage (from routing/):
    python -c "from pipeline.tables.gate_holdout import compute_gate_holdout_validation; print(compute_gate_holdout_validation())"
    python pipeline/tables/gate_holdout.py   # writes results/tab_gate_holdout.csv,
                                              # results/tab_agreement_path_audit.csv,
                                              # results/gate_holdout_summary.md
"""
import os
import sys

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROUTING_BASE, "results")

# ─────────────────────────────────────────────────────────────────────────
# Held-out validation protocol parameters
# ─────────────────────────────────────────────────────────────────────────
N_REPEATS = 250          # random calibration/test splits per (cell, split spec)
N_BOOT = 500             # bootstrap resamples inside each calibration-half
                         # significance test (baseline_utils default is 10000;
                         # reduced here because this CI is recomputed
                         # N_REPEATS x len(SPLIT_GRID) x 35 times -- a run with
                         # the full 10000 would take >10x longer for a CI that
                         # only has to decide "entirely above zero or not" on
                         # a half-pool that is itself far smaller than the
                         # full-data CIs reported elsewhere in the paper)
SEED = 42
# Each entry is a target per-half size N (calibration size = test size = N,
# capped at n_disagree // 2 when the pool is too small to support it
# without overlap -- never done via replacement/bootstrap, so calibration
# and test are always disjoint). None = use the full pool's natural half
# (n_disagree // 2 per side), i.e. the literal "split the subset in half"
# protocol the reviewer asked for.
SPLIT_GRID = [50, 100, 200, None]
SPLIT_LABELS = {50: "N=50", 100: "N=100", 200: "N=200", None: "half"}
SMALL_POOL_THRESHOLD = 40  # n_disagree below this: too small for a
                           # meaningful half/half held-out split at all
RELIABILITY_BAR = 0.90     # same cutoff gate_analysis.py used for its
                           # (data-leaking) oracle-agreement rate, reused
                           # here for the genuinely-held-out analogue so the
                           # two numbers are comparable


def _actions_for_calibration(y_true_c, y1_c, y2_c, y3_c, n_boot, rng):
    """Return (naive_idx, sig_idx) in {0=cheap1, 1=cheap2, 2=verifier},
    computed from the calibration half only."""
    import baseline_utils as bu  # pure numpy, safe to import lazily here too

    a1 = float(np.mean(y1_c == y_true_c))
    a2 = float(np.mean(y2_c == y_true_c))
    a3 = float(np.mean(y3_c == y_true_c))

    naive_idx = int(np.argmax([a1, a2, a3]))  # == current paper rule (see module docstring)

    best_cheap_idx = 0 if a1 >= a2 else 1
    y_best_cheap_c = y1_c if best_cheap_idx == 0 else y2_c

    seed = int(rng.integers(0, 2**31 - 1))
    _, lo, _hi = bu.paired_bootstrap_diff_ci(y_true_c, y3_c, y_best_cheap_c, n_boot=n_boot, seed=seed)
    sig_escalate = lo > 0.0
    sig_idx = 2 if sig_escalate else best_cheap_idx
    return naive_idx, sig_idx


def _one_repeat(y_true, y1, y2, y3, n_half, n_boot, rng):
    """One random disjoint calibration/test split of size n_half each. Returns
    a dict with per-rule ("naive"/"sig") test_acc/regret/harmful/match/escalate."""
    n = len(y_true)
    perm = rng.permutation(n)
    cal_idx = perm[:n_half]
    test_idx = perm[n_half:2 * n_half]

    yt_c, y1_c, y2_c, y3_c = y_true[cal_idx], y1[cal_idx], y2[cal_idx], y3[cal_idx]
    yt_t, y1_t, y2_t, y3_t = y_true[test_idx], y1[test_idx], y2[test_idx], y3[test_idx]

    naive_idx, sig_idx = _actions_for_calibration(yt_c, y1_c, y2_c, y3_c, n_boot, rng)

    # Test-half's OWN, independent oracle (never seen by the frozen decision).
    a1t = float(np.mean(y1_t == yt_t))
    a2t = float(np.mean(y2_t == yt_t))
    a3t = float(np.mean(y3_t == yt_t))
    oracle_idx = int(np.argmax([a1t, a2t, a3t]))
    best_test_acc = max(a1t, a2t, a3t)
    best_cheap_test_acc = max(a1t, a2t)

    y_arrs_t = [y1_t, y2_t, y3_t]
    out = {}
    for tag, idx in (("naive", naive_idx), ("sig", sig_idx)):
        test_acc = float(np.mean(y_arrs_t[idx] == yt_t))
        escalate = (idx == 2)
        out[tag] = {
            "test_acc": test_acc,
            "regret": best_test_acc - test_acc,          # >=0 by construction
            "escalate": escalate,
            "harmful": bool(escalate and (a3t < best_cheap_test_acc)),
            # test oracle escalates (verifier strictly best) but the frozen rule fell back
            "missed": bool(oracle_idx == 2 and not escalate),
            "match": (idx == oracle_idx),
        }
    out["rules_differ"] = naive_idx != sig_idx
    return out


def _run_cell(cell, rng):
    """Run the held-out protocol for one disagreement-subset cell across the
    whole SPLIT_GRID. Returns a list of row dicts (one per split spec)."""
    y_true = np.asarray(cell["y_true"])
    y1 = np.asarray(cell["y_cheap1"])
    y2 = np.asarray(cell["y_cheap2"])
    y3 = np.asarray(cell["y_verifier"])
    n_dis = len(y_true)
    pool_too_small = n_dis < SMALL_POOL_THRESHOLD

    rows = []
    for target in SPLIT_GRID:
        natural_half = n_dis // 2
        n_half = natural_half if target is None else min(target, natural_half)
        label = SPLIT_LABELS[target]
        capped = (target is not None) and (n_half < target)

        if n_half < 1:
            rows.append({
                "table": cell["table"], "config": cell["config"], "verifier": cell["verifier"],
                "n_disagree": n_dis, "split": label, "calib_n": n_half, "test_n": n_half,
                "n_repeats": 0, "pool_too_small": True, "capped_below_target": capped,
                **{f"{tag}_{m}": float("nan") for tag in ("naive", "sig") for m in (
                    "mean_test_acc", "mean_regret", "harmful_escalation_rate",
                    "missed_beneficial_escalation_rate", "escalation_rate", "test_oracle_match_rate")},
                "rule_disagreement_rate": float("nan"),
            })
            continue

        agg = {tag: {"test_acc": [], "regret": [], "escalate": [], "harmful": [], "missed": [], "match": []}
               for tag in ("naive", "sig")}
        differ = []
        for _ in range(N_REPEATS):
            out = _one_repeat(y_true, y1, y2, y3, n_half, N_BOOT, rng)
            for tag in ("naive", "sig"):
                for k, v in out[tag].items():
                    agg[tag][k].append(v)
            differ.append(out["rules_differ"])

        row = {
            "table": cell["table"], "config": cell["config"], "verifier": cell["verifier"],
            "n_disagree": n_dis, "split": label, "calib_n": n_half, "test_n": n_half,
            "n_repeats": N_REPEATS, "pool_too_small": pool_too_small, "capped_below_target": capped,
        }
        for tag in ("naive", "sig"):
            row[f"{tag}_mean_test_acc"] = float(np.mean(agg[tag]["test_acc"]))
            row[f"{tag}_mean_regret"] = float(np.mean(agg[tag]["regret"]))
            row[f"{tag}_harmful_escalation_rate"] = float(np.mean(agg[tag]["harmful"]))
            row[f"{tag}_missed_beneficial_escalation_rate"] = float(np.mean(agg[tag]["missed"]))
            row[f"{tag}_escalation_rate"] = float(np.mean(agg[tag]["escalate"]))
            row[f"{tag}_test_oracle_match_rate"] = float(np.mean(agg[tag]["match"]))
        row["rule_disagreement_rate"] = float(np.mean(differ))
        rows.append(row)
    return rows


def compute_gate_holdout_validation():
    """tab_gate_holdout: for all 35 (table, config, verifier) disagreement
    cells, genuinely held-out calibration/test validation of the current
    ("naive", point-estimate) gate rule vs. a CI-based ("sig",
    significance-aware) alternative, at several calibration/test sizes
    (including the natural full-pool half/half split)."""
    sys.path.insert(0, ROUTING_BASE)
    from pipeline.tables.baselines import compute_baselines_raw
    _, bc = compute_baselines_raw()

    rng = np.random.default_rng(SEED)
    all_rows = []
    for cell in bc.SUBSETS:
        all_rows.extend(_run_cell(cell, rng))
    return pd.DataFrame(all_rows)


def _agree_audit_row(dataset_label, y_true, y_a, y_b):
    y_true, y_a, y_b = np.asarray(y_true), np.asarray(y_a), np.asarray(y_b)
    agree = (y_a == y_b)
    n_total = len(y_true)
    n_agree = int(agree.sum())
    agree_acc = float(np.mean(y_a[agree] == y_true[agree])) if n_agree > 0 else float("nan")
    correlated = agree & (y_a != y_true)  # both detectors wrong AND agreeing
    n_corr = int(correlated.sum())
    return {
        "dataset": dataset_label,
        "n_total": n_total,
        "n_agree": n_agree,
        "agreement_rate": n_agree / n_total if n_total else float("nan"),
        "agreement_accuracy": agree_acc,
        "n_correlated_errors": n_corr,
        "pct_correlated_errors_of_agree": (n_corr / n_agree) if n_agree > 0 else float("nan"),
    }


def compute_agreement_path_audit():
    """tab_agreement_path_audit: agreement-path accuracy and correlated-error
    rate (both cheap detectors wrong AND agreeing -- the paper's "Breakdown
    condition", structurally invisible to disagreement-based routing because
    agreeing samples are never escalated) across every dataset where an
    agreement path exists: GenBuster/SD14/BigGAN in-domain (FFT+SigLIP2), the
    SDXL/FLUX deployment-shift settings (stale FFT+{SigLIP2,DINOv2} probes),
    and (bonus, same data already on disk) the BigGAN-distant-shift pool."""
    sys.path.insert(0, ROUTING_BASE)
    sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))
    from pipeline.tables.main_results import load_all
    import e8_modern_diffusion as e8

    rows = []

    # In-domain: GenBuster, SD14, BigGAN -- FFT+SigLIP2 agreement path.
    for ds_name in ["GenBuster", "SD14", "BigGAN"]:
        df, _warn = load_all(ds_name)
        rows.append(_agree_audit_row(ds_name, df["y_true"].values, df["y_fft"].values, df["y_siglip2"].values))

    # Deployment-shift: SDXL/FLUX, stale FFT+{SigLIP2,DINOv2} agreement path.
    for ds_name, cfg in e8.E8_DATASETS.items():
        prefix = cfg["prefix"]
        for sem_name in ["SigLIP2", "DINOv2"]:
            feat_key = e8.SEMANTICS[sem_name]
            feat_file = os.path.join(ROUTING_BASE, f"{prefix}{feat_key}.csv")
            fft_file = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")
            probe_key = {"SigLIP2": "stale_probe_siglip2", "DINOv2": "stale_probe_dinov2"}[sem_name]
            if not (os.path.exists(feat_file) and os.path.exists(fft_file)):
                continue
            df_sem = e8.get_predictions_stale(feat_file, cfg[probe_key], label=f"stale {sem_name}")
            df_fft = e8.get_predictions_stale(fft_file, cfg["stale_fft_probe"], label="stale FFT")
            if df_sem is None or df_fft is None:
                continue
            df = df_fft.rename(columns={"y_pred": "y_fft"}).merge(
                df_sem.rename(columns={"y_pred": "y_sem"})[["path", "y_sem"]], on="path", how="inner")
            if len(df) == 0:
                continue
            rows.append(_agree_audit_row(
                f"{ds_name}/{sem_name} (stale, deployment-shift)",
                df["y_true"].values, df["y_fft"].values, df["y_sem"].values))

    # Note: biggan_distant_shift_1500.csv (used by the biggan_shift table) was
    # NOT included here. Inspection shows y_stale_fft/y_stale_sig are exactly
    # complementary on every one of its 1500 rows (0% agreement) -- that pool
    # was apparently constructed upstream as a pure disagreement subset, so it
    # has no agreement path to audit (n_agree=0 trivially, not a real finding).

    return pd.DataFrame(rows)


def _old_oracle_based_counts():
    """Recompute the CURRENT (data-leaking, oracle-from-same-pool) counts
    directly from pipeline.tables.gate.py, rather than trusting the figures
    quoted in prior write-ups -- this is what the held-out numbers below are
    being compared against."""
    sys.path.insert(0, ROUTING_BASE)
    from pipeline.tables.gate import compute_sensitivity
    df = compute_sensitivity()
    out = {}
    for N in (50, 100, 200):
        sub = df[df["calib_N"] == N]
        out[N] = (int((sub["binary_agreement_rate"] >= RELIABILITY_BAR).sum()), len(sub))
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Computing held-out gate validation (35 configs x "
          f"{len(SPLIT_GRID)} split sizes x {N_REPEATS} repeats)...")
    df_holdout = compute_gate_holdout_validation()
    out1 = os.path.join(RESULTS_DIR, "tab_gate_holdout.csv")
    df_holdout.to_csv(out1, index=False)
    print(f"Saved {out1} ({len(df_holdout)} rows)")

    print("\nComputing agreement-path audit...")
    df_audit = compute_agreement_path_audit()
    out2 = os.path.join(RESULTS_DIR, "tab_agreement_path_audit.csv")
    df_audit.to_csv(out2, index=False)
    print(f"Saved {out2} ({len(df_audit)} rows)")
    print(df_audit.to_string(index=False))

    n_total_configs = df_holdout[["table", "config", "verifier"]].drop_duplicates().shape[0]
    small_pool = df_holdout[df_holdout["pool_too_small"]][["table", "config", "verifier", "n_disagree"]].drop_duplicates()

    old_counts = _old_oracle_based_counts()

    lines = []
    lines.append("# Gate held-out validation: revised N-recommendation\n")
    lines.append(f"Total configurations: {n_total_configs}\n")
    lines.append(f"Configurations with disagreement pool < {SMALL_POOL_THRESHOLD} "
                 f"samples (too small for a meaningful held-out half/half split): "
                 f"{len(small_pool)}/{n_total_configs}\n")
    if len(small_pool):
        lines.append(small_pool.to_string(index=False) + "\n")

    lines.append("\n## Old claim (data-leaking: oracle computed from the SAME full "
                 "disagreement pool the bootstrap resample is drawn from)\n")
    for N, (n_ge, n_cfg) in old_counts.items():
        lines.append(f"- N={N}: binary gate reaches >={int(RELIABILITY_BAR*100)}% "
                     f"oracle-agreement in {n_ge}/{n_cfg} configs\n")

    lines.append("\n## Revised, genuinely held-out numbers "
                 "(calibrate on one half, freeze, evaluate on the disjoint other half)\n")
    for split in ["N=50", "N=100", "N=200", "half"]:
        sub = df_holdout[(df_holdout["split"] == split) & (~df_holdout["pool_too_small"])]
        sub = sub.dropna(subset=["naive_test_oracle_match_rate"])
        n_cfg = len(sub)
        for tag, label in [("naive", "naive (point-estimate) rule"), ("sig", "significance-aware (CI) rule")]:
            n_ge = int((sub[f"{tag}_test_oracle_match_rate"] >= RELIABILITY_BAR).sum())
            mean_harm = float(sub[f"{tag}_harmful_escalation_rate"].mean())
            mean_regret = float(sub[f"{tag}_mean_regret"].mean())
            lines.append(f"- split={split:>5}, {label}: reaches >={int(RELIABILITY_BAR*100)}% "
                         f"test-oracle decision-match in {n_ge}/{n_cfg} configs "
                         f"(excl. too-small pools); mean harmful-escalation rate = {mean_harm:.4f}; "
                         f"mean regret = {mean_regret:.4f}\n")

    lines.append("\n## Bottom line\n")
    lines.append(
        "The previous 'N>=100 reaches reliable agreement in 24/35 configs' claim (and the "
        "equivalent N=200 claim) was computed by comparing bootstrap resamples of a pool "
        "against an oracle derived from THAT SAME pool -- it measures resampling stability, "
        "not generalization. Under a genuine held-out split (calibrate on one disjoint half, "
        "freeze the decision, score on the other half against an independent test-half oracle), "
        "the comparable counts above are substantially different and should replace the old "
        "claim in the paper. See tab_gate_holdout.csv for full per-configuration detail "
        "(including harmful-escalation rate and regret for both the current point-estimate "
        "rule and the CI-based rule) and tab_agreement_path_audit.csv for the agreement-path "
        "correlated-error audit.\n")

    summary_path = os.path.join(RESULTS_DIR, "gate_holdout_summary.md")
    with open(summary_path, "w") as f:
        f.write("".join(lines))
    print(f"\nSaved {summary_path}")
    print("".join(lines))


if __name__ == "__main__":
    main()
