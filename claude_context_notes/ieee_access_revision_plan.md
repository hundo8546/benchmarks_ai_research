# IEEE Access Revision Plan — Reviewer 1 Feedback

Paper: "On Reliable Detection of AI-Generated Multimedia Content" (Nayak & Madisetti)
Context: Multi-stage cascade detection system for AI-generated images/video. Detectors: FFT/CNNSpot (artifact), CLIP/SigLIP2/DINOv2 (semantic), Qwen2.5-VL-7B / GPT-5.5 (VLM verifier). Datasets: GenBuster-200K-mini, GenImage SD 1.4, GenImage BigGAN, AIGIBench SDXL/FLUX.

Goal of this doc: give Claude Code enough context to (1) inspect the repo for what's already computed vs. what needs new runs, and (2) help implement the generalized K-stage framework and the new experiments below.

---

## Core reviewer critique (plain summary)

The paper's cascade only makes sense if the expensive verifier (VLM) is actually better than the cheap detectors on the cases it's called on. Some tables show the verifier is *worse* than a cheap detector (Qwen vs. SigLIP2 in-domain), which undercuts the pitch. The reviewer wants this made explicit as a formal three-case framework rather than left implicit across scattered tables, wants fairer baseline comparisons, and wants missing columns added so it's clear whether gains come from routing or just from detector quality gaps.

Professor Madisetti independently suggested the same direction: generalize to a K-stage cascade rather than three ad hoc cases.

---

## The generalized framework to add

Generalize Section III from the fixed 3-detector setup to a K-stage cascade:

```
M = {f_1, ..., f_K},  costs C_1 <= C_2 <= ... <= C_K
```

Define accuracy gap between adjacent stages:
```
Delta_(i,i+1) = A_(i+1) - A_i
```

Three cases fall out of this naturally, and map onto existing results:

- **Case 1 — both cheap stages strong** (A_1, A_2 ≈ A_K): escalation rate → 0, cascade ≈ cheap-only. Maps to **BigGAN** (in-domain, FFT near-saturates).
- **Case 2 — asymmetric cheap stages** (A_1 << A_2, A_2 close to or above A_K): disagreement mostly just flags the weak detector's errors; "cascade gain" is really just detector-quality gap, not a genuine routing effect. Maps to **GenBuster / SD 1.4 in-domain** (SigLIP2 >> FFT, and SigLIP2 ≈ or > Qwen).
- **Case 3 — both cheap stages degraded** (A_1, A_2 << A_K): disagreement rate rises, verifier gain is real and attributable to routing rather than detector-picking. Maps to **SDXL / FLUX deployment shift** (stale semantic probe).

Breakdown condition (answers reviewer point 5): disagreement is only an informative routing signal when (a) f_1 and f_2 errors are weakly correlated, AND (b) at least one of A_1, A_2 is below A_K on the disagreement subset specifically. If both cheap detectors share a correlated blind spot, disagreement collapses toward zero even though both are wrong — this is the literal breakdown point. Candidate existing example: SD1.4→GenBuster cross-family cell in Table 18 (the one negative Δ).

---

## Action items mapped to reviewer points

### Writing/reframing only — no new experiments
- **(1)** Add explicit three-case formalization (Section III or new subsection in VI), tied to the K-stage math above.
- **(2)** Reframe the "cascade underperforms SigLIP2-only in-domain" result as the expected Case 2 finding, not a hidden weakness. State plainly: value proposition holds only in Case 3.
- **(7)** Consolidate/trim tables for readability.

### Pull from existing logs (check before assuming new runs needed)
- **(3)** FFT-only accuracy on SDXL/FLUX — likely already computed as a byproduct of the disagreement-rate calculation in Table VIII (need FFT predictions to know if FFT and semantic probe disagree). **Check RunPod output/logs first.**

### Requires new experiments
1. **Qwen2.5-VL-7B on SDXL/FLUX** — currently only GPT-5.5 was run as verifier on these sets. Need a new inference pass to get Qwen-only accuracy for Table 7/8 (reviewer's numbering) comparison columns.
2. **FFT-only column for cross-family deployment table (Table 18)** — need FFT accuracy per train→test generator pair (GenBuster→SD1.4, GenBuster→BigGAN, SD1.4→GenBuster, SD1.4→BigGAN, BigGAN→GenBuster, BigGAN→SD1.4). Unless FFT was already run per-pair, this needs new runs (FFT itself doesn't need retraining across families since it's not learned per-domain in the same way — but the per-pair accuracy numbers still need to be reported/collected).
3. **Fair bandit retrain (reviewer point 4)** — the existing bandit-vs-disagreement comparison (Table 15 vs Table 16) uses old CNNSpot+CLIP features for the bandit but new FFT+SigLIP2+Qwen for the disagreement rule. Retrain the bandit using FFT+SigLIP2 features (artifact confidence, semantic confidence, disagreement) so both use the same backbone. Re-run cross-dataset bandit transfer with this retrained bandit.
4. **Probe-degradation sweep (new, addresses points 1, 2, 5 empirically in one shot)** — on SD 1.4, deliberately weaken the semantic probe (SigLIP2 or CLIP) by shrinking its training fraction (e.g., 100%, 75%, 50%, 25%, 10%, 5%) while holding FFT and the verifier fixed. This sweeps A_2 from strong to weak continuously. For each point, log:
   - semantic-only accuracy (A_2)
   - FFT-only accuracy (A_1, should stay ~constant)
   - disagreement rate
   - cascade accuracy with verifier escalation
   - cascade gain over semantic-only (A_2)
   
   Plot cascade gain vs. Δ(A_2, A_verifier). This single experiment should produce a figure showing the transition from Case 2 (small/negative gain, cascade hurts) to Case 3 (positive gain, cascade helps) as the probe degrades — directly validating the K-stage framework rather than relying on cross-dataset anecdote.
5. **(Optional, strengthens point 6)** Validate a practical fallback rule: e.g., "if disagreement rate exceeds some threshold relative to the in-domain baseline (~26-28%), treat this as a signal of distribution shift and prefer stale-probe-only over cascade, or flag for retraining." Could test this rule against the existing SDXL/FLUX and cross-family results to see if it would have correctly identified when to trust vs. distrust the cascade.

---

## What to ask Claude Code to check in the repo

1. Locate the experiment scripts/logs for GenBuster, SD 1.4, BigGAN, SDXL, FLUX runs. Identify which per-sample predictions (FFT, CLIP/SigLIP2/DINOv2, Qwen, GPT-5.5) are already saved vs. need regeneration.
2. Check specifically whether FFT-only predictions exist for SDXL/FLUX and for each cross-family train→test pair (the six pairings in Table 18). If raw predictions exist, this is just an aggregation/reporting task, not a new run.
3. Check whether Qwen2.5-VL was ever run on SDXL/FLUX images (separate from GPT-5.5). If not, scope the inference job (cost/time estimate on RunPod).
4. Locate the bandit training script and confirm what features it currently uses (CNNSpot+CLIP vs FFT+SigLIP2). Scope the retrain.
5. Check whether the semantic probe training script supports an easy training-fraction parameter for the proposed degradation sweep, or whether that needs to be added.
6. Flag any RunPod cost/token budget constraints relevant to prioritizing which of the above four experiments to run first.

## Suggested priority order
1. Formalization writing (no compute cost, do immediately)
2. Repo audit (this doc's checklist) to see what's free vs. what costs compute
3. Probe-degradation sweep (highest payoff — answers points 1, 2, 5 in one experiment)
4. Bandit retrain on FFT+SigLIP2 (needed regardless, addresses point 4 directly)
5. Qwen on SDXL/FLUX + FFT cross-family columns (smaller, more mechanical additions)
6. Fallback-rule validation for point 6 (optional, strengthens practical contribution)
