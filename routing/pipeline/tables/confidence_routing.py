"""
routing.pipeline.tables.confidence_routing — joint artifact+semantic
confidence-routing baseline vs. the disagreement rule (Reviewer 1 Point 8).

See routing/pipeline/tables/CONFIDENCE_ROUTING_NOTES.md for the full
narrative writeup (formula, validation protocol, Table 11 reconciliation
finding with exact numbers).

────────────────────────────────────────────────────────────────────────
THE JOINT CONFIDENCE SIGNAL
────────────────────────────────────────────────────────────────────────
For each sample, take the calibrated P(fake) from the artifact (FFT) probe,
p_a, and from the semantic probe (SigLIP2 or DINOv2 -- in-domain, or the
SD14-trained probe applied zero-shot on SDXL/FLUX), p_s. Convert each to a
binary Shannon entropy, in nats:

    H(p) = -( p*log(p) + (1-p)*log(1-p) ),      H(p) in [0, ln(2)]

H(p) = 0  <=>  that probe is maximally certain (p in {0,1}).
H(p) = ln(2) ~= 0.693  <=>  p = 0.5, i.e. maximally uncertain.

The joint confidence signal used throughout this module is the "weakest
link" OR-combination:

    H_joint(x) = max( H(p_a(x)), H(p_s(x)) )

i.e. a sample is only as confident as its LESS confident probe. This
directly generalizes the paper's existing artifact-only entropy baseline
(`pipeline/tables/main_results.py`'s `fft_entropy`, thresholded at a FIXED
tau=0.5) to also incorporate the semantic probe's own uncertainty --
answering Reviewer 1 Point 8's complaint verbatim: "This does not establish
superiority over confidence routing generally, especially when the
semantic detector is substantially stronger than the artifact detector." A
sample is escalated to the verifier iff H_joint(x) > tau.

────────────────────────────────────────────────────────────────────────
THRESHOLD SELECTION -- a SEPARATE validation split, never the eval set
────────────────────────────────────────────────────────────────────────
In-domain (GenBuster/SD14/BigGAN): the frozen FFT/SigLIP2 probes (already
trained on `pipeline.splits.get_train_mask()`'s training pool, exactly as
for every other table in this codebase) are applied, UNMODIFIED, to a
sub-validation split carved out of that same training pool -- 50% of it,
fixed seed 42, disjoint from the canonical eval set by construction (the
whole pool is disjoint from eval). tau is set to the
(1 - p_D_val)-quantile of H_joint on that sub-validation split, where
p_D_val is the FFT-vs-SigLIP2 disagreement RATE observed on that same
sub-validation split (never on the eval set). The frozen tau is then
applied, as-is, to the canonical eval set to decide escalation.

SDXL/FLUX (e8 deployment-shift, zero-shot SD14-trained probes): these
datasets carry no train/eval split of their own -- `pipeline.splits`
explicitly excludes them (100% deployment/eval data, see its module
docstring). Calibrating tau directly on them would violate the
separate-validation requirement. Instead, tau is calibrated on SD14's OWN
sub-validation split (identical 50/50 sub-split of SD14's
`get_train_mask()` training pool, seed 42) using the SAME stale probes
that are deployed zero-shot to SDXL/FLUX
(`sd14_{siglip2,dinov2}_fftvariant_probe.pkl` / `sd14_fft_probe.pkl`, the
exact pickles `experiments/e8_modern_diffusion.py`'s `E8_DATASETS` cfg
points at). That frozen tau is then applied, untouched, to SDXL/FLUX. This
mirrors the paper's own stale-probe deployment story: calibrate on the
source domain (SD14), freeze the policy, deploy it zero-shot to the
unseen target domain (SDXL/FLUX) -- never peeking at SDXL/FLUX labels or
scores to pick tau.

────────────────────────────────────────────────────────────────────────
MATCHED ESCALATION RATE *and* MATCHED COST
────────────────────────────────────────────────────────────────────────
The cascade cost model used throughout this codebase
(`main_results.py`'s `LATENCY` / `ALWAYS_ESC_LATENCY` dicts) is

    cost = cheap_latency + escalation_rate * verifier_latency

for a FIXED pair of cheap detectors and a FIXED verifier. Matching the
escalation RATE between two rules that share the same cheap/verifier
components therefore automatically matches their COST too -- there is no
free cost parameter in this model beyond the escalation rate. This module
reports one escalation-rate-matched comparison per config and derives the
implied latency for BOTH rules from those same constants, rather than
re-deriving a separate "cost-matched" variant that would coincide with it
exactly (this equivalence is itself reported, not assumed silently).

Because tau is frozen on the validation split, the achieved eval escalation
rates of the two rules can differ. Column `rate_matching` therefore also
reports `exact_count`: escalate the k highest-H_joint eval samples, k = the
disagreement rule's escalation count (no labels used).

────────────────────────────────────────────────────────────────────────
NON-ESCALATED PREDICTION
────────────────────────────────────────────────────────────────────────
A sample the confidence rule does not escalate keeps a cheap prediction.
Two variants are reported (column `fallback`):
  semantic        the semantic probe's prediction (the stronger cheap
                  detector on every benchmark; primary row)
  more_confident  whichever probe has the lower entropy on that sample
Both coincide with the shared prediction wherever the probes agree. The
disagreement rule's non-escalated samples are all agreement-region, so its
prediction there is the same under either variant.

────────────────────────────────────────────────────────────────────────
VERIFIER LABELS
────────────────────────────────────────────────────────────────────────
In-domain: Qwen2.5-VL-7B, cached for the full eval set.
SDXL/FLUX: GPT-5.5 from the uniform (every-image) Batch API run,
results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv, which covers all
4,000 samples per benchmark. Both rules read the same labels, and with them
the disagreement rule reproduces tab_e8_results' GPT-5.5 cascade accuracy
exactly.

CPU-only: all heavy imports (torch/cv2/decord, pulled in transitively via
extractors.common) are deferred to inside
compute_confidence_routing_comparison(), not this module's top level, for
the same reason as pipeline/tables/main_results.py and baselines.py. No
GPU or API calls are made anywhere in this module -- every probability and
every verifier label used here is read from existing on-disk probes/CSVs.
"""
import os

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPS = 1e-12

IN_DOMAIN_DATASETS = ["GenBuster", "SD14", "BigGAN"]
E8_CONFIGS = [("SDXL", "SigLIP2"), ("SDXL", "DINOv2"), ("FLUX", "SigLIP2"), ("FLUX", "DINOv2")]

SUBVAL_SEED = 42
SUBVAL_FRAC = 0.5  # fraction of the probe-training pool held out as sub-val

UNIFORM_GPT55_CSV = os.path.join(ROUTING_BASE, "results", "batch_requests",
                                 "CONSOLIDATED_full_sdxl_flux.csv")
FALLBACKS = ("semantic", "more_confident")


def _fallback_pred(y_a, y_s, H_a, H_s, kind):
    if kind == "semantic":
        return np.asarray(y_s)
    return np.where(np.asarray(H_a) < np.asarray(H_s), y_a, y_s)


def _exact_count_mask(H_joint, k):
    """Escalate the k highest-H_joint samples: matches the disagreement rule's
    escalation count exactly, using no labels."""
    mask = np.zeros(len(H_joint), dtype=bool)
    mask[np.argsort(-np.asarray(H_joint), kind="stable")[:k]] = True
    return mask


def _rule_rows(y_true, y_a, y_s, y_ver, H_a, H_s, H_joint, dis_mask, tau, base, baseline_utils_mod, latency_fn):
    """One row per (rate_matching, fallback). rate_matching='val_tau' is the
    primary protocol (tau frozen on the validation split); 'exact_count' is a
    robustness check that escalates exactly as many eval samples as the
    disagreement rule. The disagreement rule is identical in every row."""
    dis_pred = np.where(dis_mask, y_ver, y_s)
    dis_acc = float(np.mean(dis_pred == y_true))
    dis_esc = float(np.mean(dis_mask))
    masks = {"val_tau": np.asarray(H_joint) > tau,
             "exact_count": _exact_count_mask(H_joint, int(np.sum(dis_mask)))}
    rows = []
    for matching, conf_mask in masks.items():
        conf_esc = float(np.mean(conf_mask))
        for kind in FALLBACKS:
            conf_pred = np.where(conf_mask, y_ver, _fallback_pred(y_a, y_s, H_a, H_s, kind))
            diff, lo, hi = baseline_utils_mod.paired_bootstrap_diff_ci(
                y_true, conf_pred, dis_pred, n_boot=10000, seed=42)
            rows.append({
                **base, "rate_matching": matching, "fallback": kind,
                "disagreement_rule_acc": dis_acc, "disagreement_rule_esc_rate": dis_esc,
                "confidence_rule_acc": float(np.mean(conf_pred == y_true)),
                "confidence_rule_esc_rate": conf_esc,
                "acc_diff_conf_minus_dis": diff, "diff_ci_lo": lo, "diff_ci_hi": hi,
                "latency_disagreement_ms": latency_fn(dis_esc),
                "latency_confidence_ms": latency_fn(conf_esc),
            })
    return rows


def _entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _subval_split(df, seed=SUBVAL_SEED, frac=SUBVAL_FRAC):
    """Deterministic 50/50 sub-split of an already-train-pool-restricted df.
    Returns (sub_train_df, sub_val_df); only sub_val is actually used for
    threshold selection, but sub_train is kept/returned for auditability."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(idx) * frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _fit_tau(h_val, disagree_val):
    """tau = the (1 - p_D_val) quantile of H_joint on the validation split,
    where p_D_val is the disagreement rate on that SAME validation split.
    Escalating when H_joint > tau then targets escalation rate p_D_val on
    the validation split by construction; the ACHIEVED rate on the eval set
    (computed by the caller) is reported alongside it, since tau is frozen
    before the eval set is touched and the two rates need not coincide
    exactly out-of-sample."""
    p_d_val = float(np.mean(disagree_val))
    tau = float(np.quantile(h_val, 1 - p_d_val)) if len(h_val) else float("nan")
    return tau, p_d_val


def _latency(ds_name, esc_rate, latency_mod, verifier_latency_key="qwen"):
    fft_lat = latency_mod.LATENCY["FFT"]
    sig_lat = latency_mod.LATENCY["SigLIP2"]
    verifier_lat = latency_mod.ALWAYS_ESC_LATENCY[ds_name] if verifier_latency_key == "qwen" else None
    if verifier_lat is None:
        return float("nan")
    return fft_lat + sig_lat + esc_rate * verifier_lat


# ────────────────────────────────────────────────────────────────────────
# In-domain: GenBuster / SD14 / BigGAN (verifier = Qwen2.5-VL-7B, full
# eval-set coverage -- no verifier-coverage gap here).
# ────────────────────────────────────────────────────────────────────────

def _load_indomain_probframe(ds_name, rb, DS_CFG):
    """Per-sample FFT/SigLIP2 probabilities + entropies + y_qwen, for the
    FULL dataset (train pool + eval set), via retrain_bandits_fft_siglip2's
    existing probe-application helpers (get_fft_frame/get_siglip2_frame) --
    reused rather than re-derived."""
    df_fft = rb.get_fft_frame(ds_name, val_paths=np.array([]))  # val_paths arg unused in get_fft_frame
    df_sig = rb.get_siglip2_frame(ds_name)
    df = df_sig.merge(df_fft, on="path", how="inner")

    cfg = DS_CFG[ds_name]
    prefix = cfg["prefix"]
    qwen_file = {"GenBuster": "qwen_preds.csv", "SD14": "sd14_qwen_preds.csv",
                 "BigGAN": "biggan_qwen_preds.csv"}[ds_name]
    df_qwen = pd.read_csv(os.path.join(ROUTING_BASE, qwen_file))
    # Left join: fitting tau needs only probe outputs, so the sub-val split must
    # not shrink to the rows that happen to have Qwen coverage.
    df = df.merge(df_qwen[["path", "y_qwen"]], on="path", how="left")

    df["H_a"] = df["entropy1"]           # FFT entropy, already H(p_fake) per get_fft_frame
    df["H_s"] = df["siglip2_entropy"]    # SigLIP2 entropy, already H(p_fake) per get_siglip2_frame
    df["H_joint"] = np.maximum(df["H_a"], df["H_s"])
    df["disagree"] = (df["y_fft"] != df["y_siglip2"]).astype(bool)
    return df


def _indomain_cell(ds_name, rb, DS_CFG, splits_mod, main_results_mod, baseline_utils_mod):
    df = _load_indomain_probframe(ds_name, rb, DS_CFG)

    train_mask = splits_mod.get_train_mask(df, ds_name, path_col="path")
    eval_mask = splits_mod.get_eval_mask(df, ds_name, path_col="path")
    pool_df = df[train_mask].reset_index(drop=True)
    eval_df = df[eval_mask].reset_index(drop=True)
    n_eval_missing_qwen = int(eval_df["y_qwen"].isna().sum())
    if n_eval_missing_qwen:
        raise ValueError(f"{ds_name}: {n_eval_missing_qwen} eval rows lack a Qwen prediction")

    sub_train, sub_val = _subval_split(pool_df)
    tau, p_d_val = _fit_tau(sub_val["H_joint"].values, sub_val["disagree"].values)

    base = {
        "regime": "indomain", "dataset": ds_name, "semantic": "SigLIP2",
        "verifier": "Qwen2.5-VL-7B", "n_eval": int(len(eval_df)),
        "n_subval": int(len(sub_val)), "n_subtrain": int(len(sub_train)),
        "tau": tau, "p_D_subval": p_d_val,
        "validation_source": ("50% sub-split of this dataset's own probe-training "
                               "pool (pipeline.splits.get_train_mask), seed=42"),
    }
    return _rule_rows(
        y_true=eval_df["y_true"].values, y_a=eval_df["y_fft"].values,
        y_s=eval_df["y_siglip2"].values, y_ver=eval_df["y_qwen"].values.astype(int),
        H_a=eval_df["H_a"].values, H_s=eval_df["H_s"].values, H_joint=eval_df["H_joint"].values,
        dis_mask=eval_df["disagree"].values, tau=tau,
        base=base, baseline_utils_mod=baseline_utils_mod,
        latency_fn=lambda esc: _latency(ds_name, esc, main_results_mod))


# ────────────────────────────────────────────────────────────────────────
# SDXL / FLUX (e8 deployment-shift, verifier = GPT-5.5, stale SD14-trained
# probes applied zero-shot). Validation split: SD14's OWN sub-val, using
# the same stale probes.
# ────────────────────────────────────────────────────────────────────────

def _load_e8_probframe(ds_name, sem_name, e8_mod):
    cfg = e8_mod.E8_DATASETS[ds_name]
    prefix = cfg["prefix"]
    feat_key = e8_mod.SEMANTICS[sem_name]
    probe_key = {"SigLIP2": "stale_probe_siglip2", "DINOv2": "stale_probe_dinov2"}[sem_name]

    feat_file = os.path.join(ROUTING_BASE, f"{prefix}{feat_key}.csv")
    fft_file = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")

    df_sem = e8_mod.get_predictions_stale(feat_file, cfg[probe_key], label=f"stale {sem_name}")
    df_fft = e8_mod.get_predictions_stale(fft_file, cfg["stale_fft_probe"], label="stale FFT")

    df = df_fft.rename(columns={"y_pred": "y_fft", "y_prob": "p_fft"}).merge(
        df_sem.rename(columns={"y_pred": "y_sem", "y_prob": "p_sem"})[["path", "y_sem", "p_sem"]],
        on="path", how="inner")
    df["disagree"] = (df["y_fft"] != df["y_sem"]).astype(bool)
    df["H_a"] = _entropy(df["p_fft"])
    df["H_s"] = _entropy(df["p_sem"])
    df["H_joint"] = np.maximum(df["H_a"], df["H_s"])

    df_gpt = pd.read_csv(UNIFORM_GPT55_CSV)
    df["y_gpt55"] = df["path"].map(dict(zip(df_gpt["path"], df_gpt["y_verifier"])))
    n_missing = int(df["y_gpt55"].isna().sum())
    if n_missing:
        raise ValueError(f"{ds_name}/{sem_name}: {n_missing} samples lack a uniform GPT-5.5 label")
    df["y_gpt55"] = df["y_gpt55"].astype(int)
    return df


def _sd14_subval_for_e8(sem_name, e8_mod, splits_mod):
    """SD14's own sub-validation split, scored with the SAME stale
    (SD14-trained, applied-zero-shot-elsewhere) probes E8_DATASETS points
    SDXL/FLUX at. This is the calibration data for tau -- SDXL/FLUX labels
    and scores are never touched."""
    cfg = e8_mod.E8_DATASETS["SDXL"]  # stale probe paths are identical across SDXL/FLUX cfgs
    probe_key = {"SigLIP2": "stale_probe_siglip2", "DINOv2": "stale_probe_dinov2"}[sem_name]
    feat_key = e8_mod.SEMANTICS[sem_name]

    sd14_prefix = "sd14_"
    sem_feat_file = os.path.join(ROUTING_BASE, f"{sd14_prefix}{feat_key}.csv")
    fft_feat_file = os.path.join(ROUTING_BASE, f"{sd14_prefix}fft_features.csv")

    df_sem = e8_mod.get_predictions_stale(sem_feat_file, cfg[probe_key], label=f"SD14 stale {sem_name}")
    df_fft = e8_mod.get_predictions_stale(fft_feat_file, cfg["stale_fft_probe"], label="SD14 stale FFT")
    df = df_fft.rename(columns={"y_pred": "y_fft", "y_prob": "p_fft"}).merge(
        df_sem.rename(columns={"y_pred": "y_sem", "y_prob": "p_sem"})[["path", "y_sem", "p_sem"]],
        on="path", how="inner")

    train_mask = splits_mod.get_train_mask(df, "SD14", path_col="path")
    pool_df = df[train_mask].reset_index(drop=True)
    pool_df["disagree"] = (pool_df["y_fft"] != pool_df["y_sem"]).astype(bool)
    pool_df["H_joint"] = np.maximum(_entropy(pool_df["p_fft"]), _entropy(pool_df["p_sem"]))

    _, sub_val = _subval_split(pool_df)
    return sub_val


def _e8_cell(ds_name, sem_name, e8_mod, splits_mod, baseline_utils_mod):
    df = _load_e8_probframe(ds_name, sem_name, e8_mod)
    sub_val = _sd14_subval_for_e8(sem_name, e8_mod, splits_mod)
    tau, p_d_val = _fit_tau(sub_val["H_joint"].values, sub_val["disagree"].values)

    # Latency: cheap stage + esc_rate * mean GPT-5.5 latency measured on the
    # disagreement-run file (the uniform batch run records no latency).
    gpt_lat = float(pd.read_csv(
        os.path.join(ROUTING_BASE, f"{e8_mod.E8_DATASETS[ds_name]['prefix']}"
                                    f"gpt55_{sem_name.lower()}_disagree_preds.csv")
    ).get("latency_ms", pd.Series([float("nan")])).mean())
    cheap_lat = e8_mod.LATENCY_FFT_MS + e8_mod.LATENCY_SEM_MS

    base = {
        "regime": "e8_deployment_shift", "dataset": ds_name, "semantic": sem_name,
        "verifier": "GPT-5.5", "n_eval": int(len(df)),
        "n_subval": int(len(sub_val)), "n_subtrain": None,
        "tau": tau, "p_D_subval": p_d_val,
        "validation_source": ("50% sub-split of SD14's own probe-training pool "
                               "(pipeline.splits.get_train_mask), scored with the SAME "
                               "stale probes deployed zero-shot on SDXL/FLUX; seed=42"),
    }
    return _rule_rows(
        y_true=df["y_true"].values, y_a=df["y_fft"].values, y_s=df["y_sem"].values,
        y_ver=df["y_gpt55"].values, H_a=df["H_a"].values, H_s=df["H_s"].values,
        H_joint=df["H_joint"].values, dis_mask=df["disagree"].values, tau=tau,
        base=base, baseline_utils_mod=baseline_utils_mod,
        latency_fn=lambda esc: cheap_lat + esc * gpt_lat)


def compute_confidence_routing_comparison():
    """Main entry point: joint artifact+semantic confidence-routing rule vs.
    the disagreement rule, at a matched escalation rate/cost, for:
      - GenBuster/SD14/BigGAN in-domain (verifier=Qwen2.5-VL-7B)
      - SDXL/FLUX x {SigLIP2,DINOv2} deployment-shift (verifier=GPT-5.5,
        the paper's positive-result configs / tab:e8_results)

    tau is always selected on a held-out validation split that is disjoint
    from the eval set used for the reported accuracy numbers -- see module
    docstring for the exact protocol per regime. Returns a DataFrame; also
    see compute_confidence_routing_comparison.__module__'s
    CONFIDENCE_ROUTING_NOTES.md for the written-up Table 11 reconciliation
    finding (a related but separate analysis, exposed via
    reconcile_table11_gap() below).
    """
    import sys
    sys.path.insert(0, ROUTING_BASE)
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))

    from extractors.common import DATASETS as DS_CFG
    import pipeline.splits as splits_mod
    import pipeline.tables.main_results as main_results_mod
    import retrain_bandits_fft_siglip2 as rb
    import baseline_utils as baseline_utils_mod
    import e8_modern_diffusion as e8_mod

    rows = []
    for ds_name in IN_DOMAIN_DATASETS:
        rows.extend(_indomain_cell(ds_name, rb, DS_CFG, splits_mod, main_results_mod, baseline_utils_mod))
    for ds_name, sem_name in E8_CONFIGS:
        rows.extend(_e8_cell(ds_name, sem_name, e8_mod, splits_mod, baseline_utils_mod))

    return pd.DataFrame(rows)


def reconcile_table11_gap():
    """Reviewer 1 Point 8, second half: reconcile the paper's reported
    Table 11 (tab:transfer) numbers -- GenBuster-trained learned-escalation-
    classifier in-domain accuracy 0.917 vs. disagreement-rule accuracy 0.898
    -- against the paper's "no measurable improvement" claim, with a proper
    paired-bootstrap CI (baseline_utils.paired_bootstrap_diff_ci).

    IMPORTANT REPRODUCIBILITY CAVEAT (see CONFIDENCE_ROUTING_NOTES.md for
    the full account): the current on-disk GenBuster data no longer
    reproduces the literal 0.917/0.898 pair quoted in the manuscript --
    GenBuster's probe-training pool now has Qwen2.5-VL-7B coverage outside
    its val set (retrain_bandits_fft_siglip2.train_bandit's internal-split
    fallback branch, documented as active when the manuscript's numbers
    were produced, is no longer triggered on the current data: there are
    now 199 train-split rows with Qwen coverage, so the STANDARD train/val
    protocol applies instead of the internal-split fallback). This function
    reports BOTH:
      (a) the CURRENT STANDARD-PROTOCOL result (full 1000-row official val
          split, classifier trained on the now-available 199 train rows),
          and
      (b) a faithful reproduction of the LEGACY documented protocol (a
          70/30 internal split of just the val-covered rows, exactly
          mirroring retrain_bandits_fft_siglip2.py's internal_split branch
          and bandit_transfer_fft_siglip2.py's GenBuster-row caveat) on
          today's (expanded) data,
    together with a paired-bootstrap 95% CI on (classifier_acc -
    disagreement_acc) for each, and states plainly whether the gap is
    distinguishable from zero.
    """
    import sys
    sys.path.insert(0, ROUTING_BASE)
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    import retrain_bandits_fft_siglip2 as rb
    import baseline_utils as baseline_utils_mod
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    ds_name, prefix, qwen_filename, val_paths_file, output_pickle = rb.DATASETS[0]
    assert ds_name == "GenBuster"
    df, val_paths = rb.load_dataset(ds_name, qwen_filename, val_paths_file)

    results = {}

    # (a) current standard protocol, via the module's own main()-equivalent logic.
    model, scaler, internal_split, eval_idx = rb.train_bandit(df, val_paths)
    if internal_split:
        eval_idx_a = eval_idx
        label_a = "internal held-out split (legacy fallback still active)"
    else:
        eval_idx_a = df[df["path"].isin(val_paths)].index.values
        label_a = "official val split (standard protocol -- fallback no longer triggered)"
    val_acc_a, val_esc_a, eval_df_a = rb.evaluate_bandit(df, eval_idx_a, model, scaler)
    dis_pred_a = np.where(eval_df_a["disagree"] == 1, eval_df_a["y_qwen"], eval_df_a["y_fft"])
    dis_acc_a = float(np.mean(dis_pred_a == eval_df_a["y_true"]))
    clf_pred_a = np.where(
        model.predict(scaler.transform(eval_df_a[rb.FULL_BANDIT_FEATURES].values)) == 1,
        eval_df_a["y_qwen"].values, eval_df_a["y_fft"].values)
    point_a, lo_a, hi_a = baseline_utils_mod.paired_bootstrap_diff_ci(
        eval_df_a["y_true"].values, clf_pred_a, dis_pred_a, n_boot=10000, seed=42)
    results["current_standard_protocol"] = {
        "label": label_a, "n": int(len(eval_df_a)),
        "classifier_acc": val_acc_a, "disagreement_acc": dis_acc_a,
        "diff_point": point_a, "diff_ci_lo": lo_a, "diff_ci_hi": hi_a,
        "distinguishable_from_zero": bool(lo_a > 0 or hi_a < 0),
    }

    # (b) legacy documented protocol: 70/30 internal split of ONLY the
    # val-covered rows, matching the branch that produced the manuscript's
    # historical numbers before GenBuster's pool/val-coverage grew.
    df_valcov = df[df["path"].isin(val_paths)].reset_index(drop=True)
    df_t = rb.compute_bandit_targets(df_valcov)
    rng = np.random.default_rng(42)
    all_idx = df_t.index.values
    n_train = int(len(all_idx) * 0.7)
    perm = rng.permutation(all_idx)
    train_idx_b, eval_idx_b = perm[:n_train], np.setdiff1d(all_idx, perm[:n_train])

    X = df_t[rb.FULL_BANDIT_FEATURES].values
    y = df_t["target"].values
    scaler_b = StandardScaler()
    X_train_b = scaler_b.fit_transform(X[train_idx_b])
    model_b = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_b.fit(X_train_b, y[train_idx_b])

    eval_df_b = df_t.loc[eval_idx_b].reset_index(drop=True)
    actions_b = model_b.predict(scaler_b.transform(eval_df_b[rb.FULL_BANDIT_FEATURES].values)).astype(int)
    clf_pred_b = np.where(actions_b == 1, eval_df_b["y_qwen"].values, eval_df_b["y_fft"].values)
    clf_acc_b = float(np.mean(clf_pred_b == eval_df_b["y_true"].values))
    dis_pred_b = np.where(eval_df_b["disagree"] == 1, eval_df_b["y_qwen"].values, eval_df_b["y_fft"].values)
    dis_acc_b = float(np.mean(dis_pred_b == eval_df_b["y_true"].values))
    point_b, lo_b, hi_b = baseline_utils_mod.paired_bootstrap_diff_ci(
        eval_df_b["y_true"].values, clf_pred_b, dis_pred_b, n_boot=10000, seed=42)
    results["legacy_internal_split_protocol"] = {
        "label": "70/30 internal split of the 1000 val-covered rows (legacy branch, reproduced on today's data)",
        "n": int(len(eval_df_b)),
        "classifier_acc": clf_acc_b, "disagreement_acc": dis_acc_b,
        "diff_point": point_b, "diff_ci_lo": lo_b, "diff_ci_hi": hi_b,
        "distinguishable_from_zero": bool(lo_b > 0 or hi_b < 0),
    }

    results["manuscript_reported"] = {
        "label": "Table 11 (tab:transfer) as printed in paper_updates2.tex -- NOT independently "
                  "reproducible from current on-disk data (see caveat in this function's docstring)",
        "classifier_acc": 0.917, "disagreement_acc": 0.898,
    }
    return results


if __name__ == "__main__":
    import sys
    df_out = compute_confidence_routing_comparison()
    out_path = os.path.join(ROUTING_BASE, "results", "tab_confidence_routing.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df_out.to_string(index=False))

    print("\n" + "=" * 70)
    print("Table 11 (tab:transfer) GenBuster-in-domain reconciliation")
    print("=" * 70)
    recon = reconcile_table11_gap()
    for key, val in recon.items():
        print(f"\n[{key}]")
        for k, v in val.items():
            print(f"  {k}: {v}")
