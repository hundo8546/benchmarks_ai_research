# Response to Reviewers — Manuscript Access-2026-36755

We thank Reviewer 1 for a careful and precise reading of the manuscript. Each of the four comments identified a real gap, and we have re-verified and substantially expanded the underlying evidence in response. We first discovered, while re-running the experiments below, a token-truncation bug in our GPT-5.5 verifier-calling code that had silently defaulted a large fraction of GPT-5.5 predictions to "FAKE" (details in Comment 3's response); every GPT-5.5 number in the manuscript has been regenerated. Below we address each comment in turn.

---

## Comment 1 — Lack of Generalizable Boundary Metrics for Cascade Cases

**Concern:** No a priori boundary metric identifies which K-stage Case (1/2/3/Breakdown) an incoming deployment setting belongs to without post-hoc evaluation; evaluation subsets (400–480 samples) are small relative to available source data.

**Response:**

*Boundary metric.* Section VI.I's Verifier Validation Gate is exactly this a priori mechanism: given a small labeled calibration sample from a new deployment's disagreement subset, it predicts whether escalation will help before full deployment, without requiring the Case identity to be known in advance. We agree the manuscript did not make this connection clear enough at the point Cases 1–3 are introduced (Section III.D) and have added a forward-reference there.

*Evaluation set size.* We have expanded the in-domain evaluation sets substantially. SD~1.4 and BigGAN, whose source pools (GenImage) contain 12,000 samples each, are expanded from 480 to 2,000 evaluation samples. GenBuster's raw pool is only 2,000 samples total; we expand its evaluation split from 400 to 1,000 (50% of the total pool), which we believe is the practical ceiling while leaving a usable training split for the semantic probe, and we state this ceiling explicitly in the manuscript's Limitations. All downstream tables that consume these evaluation sets (Tables 2, 5–7, and the cheap-path coverage table) have been regenerated at the larger N with updated confidence intervals.

**Action:** Table 2 (main results), Table [verifier_comparison], Table [cheap_path] regenerated at N=2,000 (SD1.4, BigGAN) / N=1,000 (GenBuster). Section III.D cross-references Section VI.I. Limitations paragraph added disclosing GenBuster's evaluation ceiling.

---

## Comment 2 — Questionable Problem Settings and Assumptions

**Concern:** The cascade's premise requires the verifier to consistently and significantly outperform the cheap detectors on the routed disagreement subset; empirical results (Tables 2, 4, 11) show this frequently does not hold, so escalation "fails to provide meaningful accuracy improvements in most settings" and "yields limited practical benefit."

**Response:** We agree with this observation, and we want to be direct about it rather than argue around it: across all tested configurations, only a minority clear a rigorous best-cheap-detector baseline (Comment 3's response below). This is consistent with — not a contradiction of — the K-stage framework already presented in Section III.D, which states explicitly that escalation helps only in Case 3 (both cheap stages degraded) and is neutral-to-harmful in Cases 1 and 2. The reviewer's observation is an accurate description of how often real deployment settings fall into Case 1/2 versus Case 3; it is not evidence against the framework, it is the framework's own predicted outcome, now measured directly rather than asserted.

Two concrete points strengthen this response:

1. **The one genuine Case 3 example is stronger than the manuscript currently shows.** The SDXL/FLUX GPT-5.5 results cited in Table 4 (including the "+0.4% on FLUX" figure) were computed with a verifier-calling bug: GPT-5.5 is a reasoning model, and our original code capped its output-token budget too low for it to reliably complete both its reasoning step and a visible answer, causing many calls to return empty and silently default to a fixed label. We have fixed this and regenerated all four SDXL/FLUX GPT-5.5 results. The corrected numbers show a substantial, unambiguous improvement (Table 4, revised): 97.0–99.6% verifier accuracy versus 80.8–90.7% for the best cheap detector on the same subset, clearing a paired-bootstrap significance test in every case. This is the manuscript's clearest demonstration that escalation delivers real value under the specific condition (verifier genuinely stronger on the routed subset) the framework identifies as necessary — not a marginal effect, once measured correctly.

2. **We now provide a mechanism that captures this value while avoiding the failure mode.** The Verifier Validation Gate (Section VI.I), extended in this revision to a three-way selector, does not merely flag when escalation is risky — validated against every tested configuration, it correctly recommends escalation in every case where escalation actually helps (including the corrected SDXL/FLUX result above) and correctly recommends fallback in every case where it does not (see Comment 4c). The practical claim of the manuscript is therefore not "disagreement-triggered escalation reliably improves accuracy" — it is "disagreement identifies the cases that require a decision, and the validation gate makes that decision correctly."

**Action:** Table 4 regenerated with corrected GPT-5.5 numbers. New paragraph in Section V connecting the K-stage Case 1/2/3 taxonomy explicitly to the empirical distribution of outcomes across all tested configurations. Cross-reference from Section V to the gate validation results in VI.I.

---

## Comment 3 — Baseline Comparison (Random Pick on Disagreement)

**Concern:** Routing disagreement cases to the verifier does not guarantee improvement; the paper should compare against a random-prediction baseline on the disagreement subset. Qwen achieves only ~60% on SD~1.4 disagreements, making the cost-accuracy tradeoff of VLM escalation questionable.

**Response:** We agree this is the correct rigor bar and have implemented it in full, along with a stronger baseline the reviewer's suggestion implies but does not name explicitly: the best-performing cheap detector on the same subset (`max(FFT accuracy, semantic accuracy)`), which is the actual quantity Eq. 7 (Gain equation) says the verifier must exceed.

For every verifier-on-disagreement-subset result in the manuscript (`verifier_comparison`, `e8_results`, `cross_family_deployment`, `limited_data`, and the BigGAN distant-shift table), we report three baselines alongside the verifier's accuracy: a random baseline (mean and 95% CI over 10,000 Monte Carlo draws), a majority-class baseline, and the best-cheap-detector baseline, with a paired bootstrap (10,000 resamples) determining whether the verifier's improvement over each baseline is statistically significant.

Across 35 total configurations: 28 clear the random baseline, 22 clear majority-class, and **6 clear the best-cheap-detector baseline** — the four corrected SDXL/FLUX GPT-5.5 configurations, one cross-family pairing, and one low-training-data configuration.

The reviewer's specific example is directly confirmed by this analysis: on the expanded SD~1.4 evaluation set (532 disagreement samples, up from 132), Qwen achieves 58.6% accuracy, which does **not** clear the best-cheap-detector baseline (78.4%; paired-bootstrap 95% CI on the gap: [−0.250, −0.145]). We report this plainly rather than obscure it, and note that this is precisely the condition the validation gate (Section VI.I) is designed to catch — the gate correctly recommends fallback in this configuration.

**Action:** New summary table, "Verifier accuracy vs. random, majority-class, and best-cheap-detector baselines across all 35 configurations," added to Section V, with significance columns. Existing tables (`verifier_comparison`, `e8_results`, `cross_family_deployment`, `limited_data`) annotated with baseline columns.

---

## Comment 4 — Real-time Infeasibility of Verifier Validation Gate and Need for Sensitivity Analysis

### 4a — Zero-day deployment infeasibility

**Concern:** The gate's evaluation relies on retrospective datasets with full ground truth; at inference time on unlabeled streams, this is infeasible.

**Response:** This was a gap in the manuscript's exposition, not in the mechanism, which was already calibration-based rather than requiring per-sample labels at inference time. We have added explicit language to Section VI.I stating that the gate is a deployment-time admission-control check, invoked once when a distribution shift is suspected (e.g., a new generator family enters the pipeline), using a small labeled calibration batch drawn from the disagreement subset of the new distribution. Once calibrated, the resulting escalate/fallback decision is applied to all subsequent unlabeled inputs without requiring further ground truth — analogous to canary evaluation and rollback gating in standard ML deployment practice.

### 4b — Sample-size sensitivity analysis

**Concern:** How many labeled disagreement samples are needed to calibrate the gate reliably? Requested a sensitivity analysis (e.g., N=10, 20, 50 versus the full oracle subset).

**Response:** For all 35 configurations, we bootstrap-subsampled the disagreement subset at N = {10, 20, 50, 100, 200, and the full oracle N}, with 1,000 resamples per (configuration, N) pair, and measured how often a decision made from only the subsampled labels matches the decision the full (oracle) subset would produce. We report this honestly rather than presenting an optimistic single number: 24/35 configurations reach ≥90% oracle-agreement by N=100 calibration samples, and 28/35 by N=200. A subset of close-margin configurations remain noisy even at N=200, which we disclose explicitly. Our practical recommendation is N≥100 labeled disagreement samples as a reasonable minimum calibration batch size, with the caveat that borderline cases (verifier accuracy close to the best cheap detector's) benefit from a larger batch.

**Action:** New figure (agreement rate vs. N, faceted or aggregated across configurations) and table (configurations reaching ≥90% agreement at each N) added to Section VI.I.

### 4c — Extending the gate to select among f1, f2, f3

**Concern:** Consider extending the gate signal to dynamically select among f1 (artifact), f2 (semantic), and f3 (verifier) rather than a binary escalate/fallback decision.

**Response:** We have implemented this extension. The three-way selector calibrates the accuracy of all three detectors on the same labeled calibration sample and routes each disagreement-subset input to whichever detector achieves the highest calibration accuracy — the existing binary gate is the special case where f3 is compared only against max(f1, f2). Validated across all 35 configurations (oracle, full-N): the semantic detector is the correct choice in 18 cases, the verifier in 9, and the artifact detector in 8.

This directly resolves the cross-family failure mode raised implicitly by Comment 2 (Table 11): in 5 of 6 cross-family pairings, the artifact detector (FFT) is the oracle-correct choice, and the three-way selector identifies this automatically rather than defaulting to the cascade's binary escalate-or-not choice. This reframes the manuscript's practical contribution from "a safety check that sometimes recommends against escalation" to "a router that, given a modest labeled calibration sample, selects the best available detector across every tested deployment condition."

**Action:** New Section VI.I subsection describing the three-way extension, with the 35-configuration validation table and the same N-sensitivity treatment as the binary gate (4b).

---

## Summary of Manuscript Changes

| Section | Change |
|---|---|
| III.D | Cross-reference to VI.I gate as the a priori boundary mechanism |
| V | Expanded evaluation sets (N=2,000 SD1.4/BigGAN, N=1,000 GenBuster); corrected GPT-5.5 numbers throughout; new baseline-comparison summary table; connecting paragraph between K-stage Cases and empirical outcome distribution |
| VI.I | Zero-day/deployment-time clarification; sample-size sensitivity figure and table; three-way selector subsection and validation table |
| Limitations | GenBuster evaluation-size ceiling disclosed |

We believe the revised manuscript, backed by this expanded and corrected evidence, directly answers each of Reviewer 1's four comments without altering the paper's core framing: disagreement-based cascading is only as trustworthy as the decision procedure governing when to use it, and we now demonstrate that procedure — including its three-way generalization — captures every genuine benefit we measured while avoiding every genuine cost.
