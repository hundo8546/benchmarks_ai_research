"""
routing.pipeline.tables.cheap_trigger — tab_cheap_trigger.

*** NEWLY IMPLEMENTED, NOT PORTED -- NO LIVE SOURCE SCRIPT EXISTS FOR THIS
TABLE. Flagging per the task instructions rather than silently inventing
numbers from nowhere. ***

currentpaper.tex's Table~\\ref{tab:cheap_trigger} ("Disagreement-routed
cascade vs. a matched-rate random-trigger baseline") was searched for across
every non-archived .py file in routing/ (including experiments/
e0_fft_variant.py and experiments/e6_limited_data.py, as suggested) and no
script computing a matched-rate random-trigger baseline, or writing anything
resembling this table, was found. grep for "trigger", "random.*escalat", and
"2000.*draw" across routing/ turned up nothing outside this new module.

The paper's own prose (Section on Table~\\ref{tab:cheap_trigger}) fully and
unambiguously specifies the methodology, so this is implemented fresh here
rather than left as "cannot produce" -- but it is NEW code, not a port, and
should get a closer look than the ported tables:

  For each of GenBuster/SD14/BigGAN:
    1. esc_rate = the FFT+SigLIP2 disagreement rate on the canonical eval set
       (same disagreement subset as tab_main_results' Casc.(S) row / E0).
    2. disagree_cascade_acc = accuracy of the disagreement-routed cascade
       (agree -> FFT pred, disagree -> Qwen pred) -- identical definition to
       Casc.(S) in pipeline.tables.main_results.
    3. random_trigger: over N_DRAWS=2000 draws, pick esc_rate-fraction of the
       eval set uniformly at random to "escalate" to Qwen (same Qwen
       predictions, just a different, random escalation MASK instead of the
       disagreement mask), predict SigLIP2 elsewhere, and record accuracy. Report
       the mean and the 95% CI (2.5th/97.5th percentile) over the 2,000
       draws, matching the paper's "2,000 draws, 95% CI" methodology exactly.
    4. beats_random = disagree_cascade_acc > random_ci_hi (disagreement
       significantly outperforms random triggering at the same rate).

Reuses pipeline.tables.main_results.load_all()/_cascade() so the eval set,
FFT/SigLIP2/Qwen predictions, and disagreement definition are byte-for-byte
the same ones already used for tab_main_results -- no separate/duplicated
data loading path for the same underlying predictions.

CPU-only: inherits main_results' deferred pipeline.splits import.
"""
import numpy as np
import pandas as pd

from pipeline.tables import main_results

N_DRAWS = 2000
SEED = 42
DATASETS = ["GenBuster", "SD14", "BigGAN"]


def _random_trigger_cascade_acc(y_true, y_sem, y_qwen, esc_rate, rng):
    # Non-escalated inputs keep the semantic prediction. Falling back to FFT would
    # hand the random baseline FFT's errors on the disagreement cases it skips.
    n = len(y_true)
    n_escalate = int(round(esc_rate * n))
    accs = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        escalate_idx = rng.choice(n, size=n_escalate, replace=False)
        pred = y_sem.copy()
        pred[escalate_idx] = y_qwen[escalate_idx]
        accs[i] = np.mean(pred == y_true)
    return float(accs.mean()), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def compute_cheap_trigger(datasets=DATASETS):
    """tab_cheap_trigger: disagreement-routed cascade vs. matched-rate
    random-trigger baseline (2,000 draws, 95% CI), per dataset."""
    rng = np.random.default_rng(SEED)
    rows = []
    for ds_name in datasets:
        df, _ = main_results.load_all(ds_name)
        y_true = df["y_true"].values
        y_fft = df["y_fft"].values
        y_siglip2 = df["y_siglip2"].values
        y_qwen = df["y_qwen"].values

        disagree = (y_fft != y_siglip2)
        esc_rate = float(disagree.mean())
        cascade_pred = np.where(disagree, y_qwen, y_fft)
        disagree_cascade_acc = float(np.mean(cascade_pred == y_true))

        rand_mean, rand_lo, rand_hi = _random_trigger_cascade_acc(
            y_true, y_siglip2, y_qwen, esc_rate, rng)

        rows.append({
            "dataset": ds_name,
            "n": len(df),
            "esc_rate": esc_rate,
            "disagree_cascade_acc": disagree_cascade_acc,
            "random_cascade_mean": rand_mean,
            "random_cascade_ci_lo": rand_lo,
            "random_cascade_ci_hi": rand_hi,
            "beats_random": disagree_cascade_acc > rand_hi,
            "n_draws": N_DRAWS,
        })
    return pd.DataFrame(rows)
