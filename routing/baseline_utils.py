"""
Shared baseline-comparison utilities for Reviewer 1 Comment 3 (random-pick /
majority-class / best-cheap-detector baselines on the disagreement subset).

See claude_context_notes/resubmissionplan.md Section 3.3.
"""
import numpy as np


def random_baseline(y_true, n_draws=10000, seed=42):
    """Predict each sample fake w.p. 0.5, averaged over n_draws. Returns
    (mean_acc, ci_lo, ci_hi) — a 95% CI over Monte Carlo draws, not over
    resampling the dataset."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    accs = np.empty(n_draws)
    for i in range(n_draws):
        y_rand = (rng.random(n) < 0.5).astype(int)
        accs[i] = np.mean(y_rand == y_true)
    return float(accs.mean()), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def majority_class_baseline(y_true):
    """Deterministic majority-class accuracy on this subset."""
    y_true = np.asarray(y_true)
    p1 = float(np.mean(y_true))
    return max(p1, 1.0 - p1)


def best_cheap_detector_baseline(y_true, y_cheap1, y_cheap2, name1="cheap1", name2="cheap2"):
    """max(acc(cheap1), acc(cheap2)) on this subset. Returns (acc, source_name)."""
    acc1 = float(np.mean(np.asarray(y_cheap1) == np.asarray(y_true)))
    acc2 = float(np.mean(np.asarray(y_cheap2) == np.asarray(y_true)))
    return (acc1, name1) if acc1 >= acc2 else (acc2, name2)


def paired_bootstrap_diff_ci(y_true, y_a, y_b, n_boot=10000, seed=42):
    """95% CI on (acc(y_a) - acc(y_b)) resampling the n samples with
    replacement, consistent with the paired-bootstrap convention already
    used elsewhere in the paper (e.g. cross_family_deployment table)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    n = len(y_true)
    correct_a = (y_a == y_true).astype(float)
    correct_b = (y_b == y_true).astype(float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    point = float(correct_a.mean() - correct_b.mean())
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare_verifier_to_baselines(y_true, y_verifier, y_cheap1, y_cheap2,
                                    cheap1_name="cheap1", cheap2_name="cheap2",
                                    n_draws=10000, n_boot=10000, seed=42):
    """Full baseline comparison for one (dataset, verifier) cell. Returns a
    flat dict suitable for a DataFrame row."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    verifier_acc = float(np.mean(np.asarray(y_verifier) == y_true))

    rand_mean, rand_lo, rand_hi = random_baseline(y_true, n_draws=n_draws, seed=seed)
    maj_acc = majority_class_baseline(y_true)
    best_cheap_acc, best_cheap_src = best_cheap_detector_baseline(
        y_true, y_cheap1, y_cheap2, cheap1_name, cheap2_name)

    # Fake predictor arrays for paired bootstrap against deterministic baselines
    p1 = float(np.mean(y_true))
    maj_pred_val = 1 if p1 >= 0.5 else 0
    y_majority = np.full(n, maj_pred_val)
    y_best_cheap = np.asarray(y_cheap1) if best_cheap_src == cheap1_name else np.asarray(y_cheap2)

    _, maj_lo, maj_hi = paired_bootstrap_diff_ci(y_true, y_verifier, y_majority, n_boot=n_boot, seed=seed)
    _, bc_lo, bc_hi = paired_bootstrap_diff_ci(y_true, y_verifier, y_best_cheap, n_boot=n_boot, seed=seed)

    return {
        "n": n,
        "verifier_acc": verifier_acc,
        "random_mean": rand_mean,
        "random_ci_lo": rand_lo,
        "random_ci_hi": rand_hi,
        "clears_random": verifier_acc > rand_hi,
        "majority_acc": maj_acc,
        "verifier_minus_majority_ci_lo": maj_lo,
        "verifier_minus_majority_ci_hi": maj_hi,
        "clears_majority": maj_lo > 0,
        "best_cheap_acc": best_cheap_acc,
        "best_cheap_source": best_cheap_src,
        "verifier_minus_best_cheap_ci_lo": bc_lo,
        "verifier_minus_best_cheap_ci_hi": bc_hi,
        "clears_best_cheap": bc_lo > 0,
    }
