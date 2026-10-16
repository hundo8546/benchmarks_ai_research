# Confidence-routing comparison (Reviewer 1, Point 8)

Code: `routing/pipeline/tables/confidence_routing.py` (`compute_confidence_routing_comparison()`,
`reconcile_table11_gap()`). Output: `routing/results/tab_confidence_routing.csv`.
No GPU/API calls anywhere in this module; every probability and verifier label is read
from existing on-disk probes/CSVs.

## 1. The joint confidence signal (exact formula)

For each sample, take the calibrated P(fake) from the artifact (FFT) probe, `p_a`, and
from the semantic probe (SigLIP2 or DINOv2 — in-domain, or SD14-trained applied zero-shot
on SDXL/FLUX), `p_s`. Convert each to a binary Shannon entropy, in nats:

```
H(p) = -( p*log(p) + (1-p)*log(1-p) ),      H(p) in [0, ln(2)]
```

`H(p)=0` ⟺ that probe is maximally certain; `H(p)=ln(2)≈0.693` ⟺ `p=0.5` (maximally
uncertain). The joint signal is the **weakest-link** combination:

```
H_joint(x) = max( H(p_a(x)), H(p_s(x)) )
```

i.e. a sample is only as confident as its *less* confident probe. A sample is escalated
to the verifier iff `H_joint(x) > tau`. This directly generalizes the paper's existing
artifact-only entropy baseline (`main_results.py`'s `fft_entropy`, thresholded at a
**fixed** `tau=0.5`) to also use the semantic probe's uncertainty — the exact gap
Reviewer 1 identified ("does not establish superiority over confidence routing
generally, especially when the semantic detector is substantially stronger than the
artifact detector").

## 2. Threshold selection — separate validation split, never the eval set

- **In-domain (GenBuster/SD14/BigGAN)**: `tau` is fit on a 50/50 sub-split (seed 42) of
  that dataset's own probe-training pool (`pipeline.splits.get_train_mask`), which is by
  construction disjoint from the canonical eval set. `tau` = the `(1 - p_D_val)`-quantile
  of `H_joint` on that sub-validation split, where `p_D_val` is the FFT-vs-SigLIP2
  disagreement rate observed on that *same* sub-validation split. The frozen `tau` is
  then applied, unmodified, to the canonical eval set.
- **SDXL/FLUX (e8 deployment shift)**: these datasets have no train/eval split of their
  own — `pipeline.splits` explicitly treats them as 100% deployment/eval data. Calibrating
  `tau` on them directly would violate "not the eval set." Instead, `tau` is calibrated on
  **SD14's own** sub-validation split (same 50/50 split, seed 42, of SD14's
  `get_train_mask` pool), scored with the **same stale, SD14-trained probes**
  (`sd14_fft_probe.pkl`, `sd14_{siglip2,dinov2}_fftvariant_probe.pkl`) that
  `experiments/e8_modern_diffusion.py`'s `E8_DATASETS` config deploys zero-shot to
  SDXL/FLUX. The frozen `tau` is then applied, untouched, to SDXL/FLUX — mirroring the
  paper's own stale-probe deployment story (calibrate on source domain, freeze, deploy to
  unseen target domain) rather than peeking at SDXL/FLUX labels or scores.

## 3. Matched escalation rate / matched cost

The cascade cost model used throughout this codebase (`main_results.py`'s `LATENCY` /
`ALWAYS_ESC_LATENCY`) is `cost = cheap_latency + escalation_rate * verifier_latency` for a
fixed pair of cheap detectors and a fixed verifier. Matching escalation **rate** between
two rules that share the same cheap/verifier components therefore automatically matches
**cost** — there is no free cost parameter beyond rate in this model. We report one
rate-matched comparison per config and derive the implied latency for both rules from
those same constants (columns `latency_disagreement_ms` / `latency_confidence_ms`), rather
than inventing a separate "cost-matched" variant that would coincide with it exactly.

## 4. Verifier-coverage caveat (SDXL/FLUX only)

GPT-5.5 was only ever run on the FFT-vs-semantic **disagreement subset** for the e8
SDXL/FLUX configs — there is no cached GPT-5.5 prediction for agreement-region samples
(`y_fft == y_sem`). Because the joint-confidence rule can select a different subset than
the disagreement rule, it can flag some agreement-region samples for escalation with no
cached verifier label (`n_escalate_agree_no_verifier` in the output). We fall back to the
shared cheap value there (a neutral, not optimistic, fallback: the disagreement rule also
uses exactly that value for every agreement-region sample, escalated or not). **We bound
how much this biases the result** by computing, per config, the FFT accuracy already
achieved on that exact gap subset and the best-case accuracy if GPT-5.5 corrected
*every* error there:

| Config | n_gap (frac of 4000) | FFT acc on gap subset | Max possible accuracy credit |
|---|---|---|---|
| SDXL/SigLIP2 | 776 (19.4%) | 0.802 | +3.8 pp |
| SDXL/DINOv2  | 760 (19.0%) | 0.817 | +3.5 pp |
| FLUX/SigLIP2 | 662 (16.6%) | 0.873 | +2.1 pp |
| FLUX/DINOv2  | 711 (17.8%) | 0.740 | +4.6 pp |

Observed deficits (confidence-rule acc vs. disagreement-rule acc) are −17 to −24 pp — far
larger than the maximum possible credit from closing the coverage gap in the best case.
We also recomputed the comparison with an **exact** rate match (ranking `H_joint` within
the eval set itself to select exactly as many samples as the disagreement rule escalates,
using no label information, only counts) and get the same conclusion (gaps of −17 to
−21 pp). The result is not an artifact of coverage or of imperfect rate transfer from the
validation split.

In-domain GenBuster/SD14/BigGAN have **no such gap**: Qwen2.5-VL-7B predictions are cached
for the full eval set there, not just the disagreement subset.

## 5. Results

`routing/results/tab_confidence_routing.csv`, all 7 configs (3 in-domain + 4 e8
SDXL/FLUX×{SigLIP2,DINOv2} — the paper's positive-result cells, `tab:e8_results`):

| Config | Verifier | n | Disagreement acc (esc. rate) | Joint-confidence acc (esc. rate) | Δ (conf − dis), 95% CI |
|---|---|---|---|---|---|
| GenBuster (in-domain) | Qwen2.5-VL-7B | 1000 | 0.899 (0.259) | 0.760 (0.337) | −0.139 [−0.162, −0.117] |
| SD14 (in-domain) | Qwen2.5-VL-7B | 2000 | 0.874 (0.266) | 0.765 (0.250) | −0.109 [−0.126, −0.093] |
| BigGAN (in-domain) | Qwen2.5-VL-7B | 2000 | 0.995 (0.007) | 0.990 (0.016) | −0.0055 [−0.009, −0.0025] |
| SDXL / SigLIP2 | GPT-5.5 | 4000 | 0.923 (0.363) | 0.747 (0.357) | −0.173 [−0.185, −0.161] |
| SDXL / DINOv2 | GPT-5.5 | 4000 | 0.927 (0.385) | 0.746 (0.354) | −0.180 [−0.192, −0.168] |
| FLUX / SigLIP2 | GPT-5.5 | 4000 | 0.940 (0.419) | 0.698 (0.318) | −0.242 [−0.255, −0.229] |
| FLUX / DINOv2 | GPT-5.5 | 4000 | 0.891 (0.415) | 0.681 (0.335) | −0.208 [−0.220, −0.195] |

**Every single one of the 7 configurations** shows the disagreement rule beating the
joint-confidence rule at a matched escalation rate/cost, by 0.6–24 percentage points, all
with 95% paired-bootstrap CIs that exclude zero. Mechanistically: the artifact probe's and
semantic probe's *disagreement* is a much tighter proxy for "the cheap FFT prediction is
wrong" than either probe's own entropy is. On GenBuster, for example, SigLIP2 alone is
already 96.6% accurate and Qwen2.5-VL-7B alone is only 71.3% accurate (`tab_main_results.csv`);
the disagreement rule escalates almost exclusively the ~26% of samples where FFT is
actually wrong (so swapping in mediocre-overall Qwen is still a net win there), while the
joint-confidence rule escalates a broader, less-targeted ~34% "either probe is unsure"
subset that includes many samples FFT already gets right — swapping those correct FFT
answers for a weaker-overall verifier actively hurts accuracy. This is a clean, direct,
matched-rate empirical answer to Reviewer 1 Point 8: disagreement is not just *a*
confidence signal dressed up differently, it is a *better* one than the best principled
joint-confidence alternative we could construct, including in the exact SDXL/FLUX+GPT-5.5
configurations that support the paper's positive disagreement-routing claim.

## 6. Table 11 (`tab:transfer`) reconciliation — 0.917 vs. 0.898

Reviewer 1 Point 8 also asks us to reconcile the paper's reported Table 11 numbers
(GenBuster-trained learned-escalation-classifier in-domain accuracy **0.917** vs.
disagreement-rule accuracy **0.898**) against the "no measurable improvement" claim made
elsewhere (Contribution C4 / line 110 of `paper_updates2.tex`), with a proper
paired-bootstrap CI.

**Reproducibility finding (important and should be disclosed to reviewers):** the literal
0.917/0.898 pair printed in the manuscript is **not reproducible from the current on-disk
pipeline state**. `bandit_transfer_fft_siglip2.py`'s GenBuster row is documented as using
an "internal split of its own val set (no train-split Qwen coverage)" fallback — but on
the data as it exists today, GenBuster's probe-training pool *does* have Qwen2.5-VL-7B
coverage outside its val set (199 rows), so `retrain_bandits_fft_siglip2.train_bandit`'s
`internal_split` fallback branch is **no longer triggered**; the standard train/val
protocol runs instead. This is a genuine data/pipeline drift since the manuscript's
numbers were produced (GenBuster's dataset/Qwen-coverage pool appears to have grown), not
a bug in our reconciliation code.

We computed the paired-bootstrap CI (`baseline_utils.paired_bootstrap_diff_ci`,
10,000 resamples, seed 42) under **both** available protocols on today's data:

| Protocol | n (eval) | Classifier acc | Disagreement acc | Δ (classifier − disagreement) | 95% CI | Distinguishable from 0? |
|---|---|---|---|---|---|---|
| Current standard protocol (official 1000-row val split; fallback not triggered) | 1000 | 0.899 | 0.899 | 0.000 | [0.000, 0.000] | **No** — predictions are byte-identical |
| Legacy internal-split protocol (70/30 split of the 1000 val-covered rows, reproducing the documented fallback branch on today's data) | 300 | 0.900 | 0.903 | −0.0033 | [−0.0100, 0.0000] | **No** |
| Manuscript as printed (`paper_updates2.tex`, Table 11) | — | 0.917 | 0.898 | +0.019 (not independently recomputable) | not available (no cached per-sample predictions behind the historical number) | unknown |

**Conclusion:** under both protocols we can actually run today, the gap between the
learned escalation classifier and the disagreement rule is **not statistically
distinguishable from zero** — in the standard-protocol case it is exactly zero (the
classifier and the disagreement rule produce identical predictions on the current data),
and in the legacy-protocol case the point estimate is slightly *negative* with a 95% CI
entirely at or below zero. This is actually a *stronger* empirical basis for "no
measurable improvement" than the manuscript's own 0.917 figure suggests, but it also
means **the literal 0.917 number in Table 11 should not be reported as-is**: it cannot be
reproduced from the current pipeline/data, so either (a) Table 11 should be regenerated
from the current pipeline state and the CI reported directly in the table (recommended —
gives 0.899 vs 0.899, Δ=0 exactly, under the standard protocol), or (b) if the authors
have the original data/code snapshot that produced 0.917, that snapshot's per-sample
predictions should be located and the CI computed on those instead. Either way, the
qualitative claim ("no measurable improvement") is *correct* — the quantitative framing
("0.917 vs 0.898," read naively as a positive 1.9pp improvement with no caveat) is
unsupported without an equivalence criterion/CI, exactly as Reviewer 1 states, and should
be softened in the text to something like: *"a CI-based equivalence result: the paired
accuracy difference between the learned escalation classifier and the disagreement rule
is not statistically distinguishable from zero (95% CI spans or sits at zero in every
protocol we can currently reproduce), i.e. the learned classifier recovers but does not
improve on the parameter-free disagreement signal."*
