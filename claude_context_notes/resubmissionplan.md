# Resubmission Plan — IEEE Access Manuscript Access-2026-36755

**Status:** Reviewer 2 fully satisfied. Reviewer 1 answered "no" on both
contribution and technical soundness. One resubmission opportunity remains.
This plan must fully resolve R1's four comments or the paper is rejected
outright with no further chances.

---

## 1. Root Cause Diagnosis

Before listing tasks, it's worth being precise about *why* R1 said no, because
the fix is not purely experimental.

R1 read this as a **detection paper** that proposes a cascade, and evaluated
it as one. Under that reading, the paper's own results say the cascade loses
to a well-trained semantic probe in-domain (Table 2), loses to FFT-only in
cross-family transfer (Table 11, 5/6 pairings), and loses to the stale probe
under the BigGAN distant shift (Table 12, all 3 verifiers). Only one table
(SDXL/FLUX with GPT-5.5) shows a win. R1's conclusion — "escalation fails to
provide meaningful accuracy improvements in most settings" — is a *correct*
reading of the paper as currently framed. This is not a misunderstanding to
argue away; it's an accurate diagnosis of a structural weakness in how the
contribution is presented.

**The fix has two parts that must both happen:**

1. **Reframe the contribution.** The paper is not "a cascade that improves
   detection." It is "a decision procedure (the validation gate) for
   determining *when* cascading helps, evidenced by a systematic mapping of
   the conditions under which cascading helps, hurts, or is neutral." The
   negative results (Case 1, Case 2, distant shift) are not embarrassing
   side findings — they are the *evidence base* that motivates why a gate is
   necessary. Right now they read as failures of the proposed method because
   the proposed method is presented as the cascade, not the gate.

2. **Strengthen the evidence with new experiments** that directly answer
   R1's four comments: a boundary metric / larger evaluation, a rigorous
   random-baseline comparison, and a feasibility-plus-sensitivity treatment
   of the gate.

Neither part alone is sufficient. Reframing without new evidence looks like
spin. New evidence without reframing still reads as "a method that mostly
doesn't work."

---

## 2. Structural Reframe (do this first — it determines what the new
   experiments need to show)

### 2.1 Title
Current: *"On Reliable Detection of AI-Generated Multimedia Content"*
This promises a detection paper and primes the reviewer to benchmark against
detectors — exactly the reading that produced comment 2.

Proposed: something in the family of *"When to Escalate: A Validation Gate
for Selective Verifier Invocation in AI-Generated Media Detection Cascades"*
or *"Disagreement-Gated Verification: Knowing When Not to Cascade in
AI-Generated Media Detection."* The title should foreground the *decision
procedure*, not the cascade itself.

### 2.2 Abstract
Rewrite so the throughline is: (1) disagreement identifies hard samples
reliably, (2) whether escalating on those samples helps is *conditional* and
this condition is formalized (Eq. 7 / K-stage cases), (3) the paper's main
contribution is a validation gate that predicts, before deployment and
without knowing the shift type, whether escalation will help — and this gate
is shown to be accurate across a wide range of conditions including ones
where escalation actively hurts. The current abstract already gestures at
the gate in its last sentence; it needs to be the headline, not the coda.

### 2.3 Contributions list (Section I)
Reorder so C5 (the gate) becomes **C1**. Reframe C3/C4 (the SDXL/FLUX
recovery and bandit convergence) as supporting evidence for *why a gate is
needed and why it can be simple*, not as independent headline claims.
Add a new contribution corresponding to the random/majority-class baseline
validation (see §3.3) and the sensitivity analysis (see §3.4) — these make
the gate itself rigorously validated rather than asserted.

### 2.4 Section III.D (K-stage formulation)
Move the validation gate's *definition* here, immediately after Eq. 7. The
three cases and the gate are two views of the same underlying quantity
(Eq. 7's sign) — present them together, not 20 pages apart. Section III.D
should end with: "Sections III.D and VI.I together specify the decision
procedure. The remainder of the paper (Section V) provides the empirical
mapping of real conditions to these cases, and validates the gate's ability
to distinguish them without oracle knowledge of which case applies."

### 2.5 Results/Discussion reorganization
Currently the negative results are scattered and read as accumulating
failures. Restructure Section V/VI so each negative result is explicitly
introduced as "a test case for the gate" rather than "a limitation of the
cascade." E.g., instead of "escalation is significantly harmful for all
three tested verifiers" as a standalone finding, frame it as "this is
precisely the condition the gate is designed to catch; §[gate result]
confirms the gate correctly recommends fallback here."

---

## 3. Point-by-Point Technical Plan

### 3.1 Comment 1 — No a priori boundary metric; small eval sets

**What's needed:**
- Explicit statement that the gate (moved forward per §2.4) *is* the a
  priori mechanism — it does not require knowing which Case applies, only a
  small labeled calibration sample.
- Address "a priori" precisely: the gate is evaluated *before full
  deployment* using a calibration sample, but it is not zero-shot / label-free.
  Say this explicitly rather than let the reviewer infer ambiguity.
- **Expand in-domain evaluation sets.** Current: 400 (GenBuster), 480 (SD1.4),
  480 (BigGAN). GenImage provides 6,000/6,000 for SD1.4 and BigGAN — there is
  no excuse for not using more of it. Target: ≥2,000 samples per benchmark
  where source data allows (GenBuster's raw pool is smaller — max out
  whatever the held-out split allows, document the ceiling honestly).
- Re-run: Table 2 (main results), Table 5-7 (agreement/escalation
  breakdown), Table cheap-path (agreement coverage) at the larger N. All
  downstream tables that consume these evaluation sets need to be
  regenerated with updated confidence intervals.
- Consider adding a note that GenBuster's real ceiling (limited by dataset
  scale) is disclosed, since SD1.4/BigGAN are less constrained.

**Deliverable:** Updated Tables 2, 5, 6, 7, cheap-path coverage table, all
with N ≥ 2,000 where feasible and tightened CIs. A short new paragraph in
Limitations acknowledging the *remaining* ceiling on GenBuster if it can't
reach 2,000.

### 3.2 Comment 2 — Contradictory results undermine the premise

**What's needed:** This is resolved primarily by the reframe (§2), but two
concrete additions strengthen it:

- Add an explicit "roadmap of results" paragraph early in Section V or at
  the start of Section VI stating plainly: "Table 2 shows escalation does
  not help in-domain (Case 1/2). Table 11 shows it does not help under
  cross-family transfer (Case 2). Table 12 shows it does not help under
  distant GAN shift (Case 2, even with a frontier verifier). Table 4 shows
  it *does* help under moderate diffusion-family shift with a strong
  verifier (Case 3). This spread of outcomes is not a weakness of the
  disagreement signal — it is the reason a validation gate is necessary; a
  system that always escalates on disagreement, or one that never does,
  would be wrong in a large fraction of these conditions." This paragraph
  should appear before the detailed tables so the reviewer reads every
  subsequent negative result as expected/diagnostic rather than surprising.
- Directly answer the FLUX/GPT-5.5 "+0.4%, massive cost" critique: add the
  random/majority baseline (§3.3) result for that exact cell so the reviewer
  can see whether +0.4% is meaningful relative to chance, and be honest in
  prose if it is not — in which case say so and let the gate (calibrated on
  that cell) recommend fallback, and show that it does. If the gate doesn't
  currently catch this, that's an important thing to fix before resubmission,
  not to paper over.

### 3.3 Comment 3 — Random-baseline comparison on disagreement subset

**This is the highest-leverage new experiment. Do it first, before anything
else, because the result determines how strong a claim the rest of the
paper can make.**

For every verifier/dataset/backbone cell currently reporting verifier
accuracy on a disagreement subset (Tables: verifier_comparison, e8_results,
biggan_shift, cross_family_deployment, limited_data), compute two baselines
on the *same* subset:
1. **Random baseline:** predict each sample as fake with probability 0.5,
   averaged over many draws (e.g., 10,000), report mean + 95% CI.
2. **Majority-class baseline:** predict the majority class of the subset,
   single deterministic number.
3. **Best-cheap-detector baseline:** max(FFT accuracy, semantic accuracy) on
   that same subset — this is arguably the *real* bar, since it's what
   Eq. 7 says the verifier must clear, and it's stronger than random or
   majority in most cells.

Report all three next to verifier accuracy in every relevant table. Add a
single summary table: "Verifier accuracy vs. random, majority-class, and
best-cheap-detector baselines across all N configurations," with a column
stating whether the verifier significantly clears each baseline (paired
bootstrap, consistent with the existing bootstrap methodology already used
elsewhere in the paper).

**Expected outcomes to plan for:**
- GPT-5.5 on SDXL/FLUX likely clears all three baselines by a wide margin —
  use this as the clean positive case.
- Qwen at ~60-64% likely clears random/majority but may not clearly clear
  best-cheap-detector in several cells — this is consistent with the
  existing Case 2 narrative and should be reported plainly.
- Qwen2.5-VL-72B on BigGAN distant shift (46.7%) may fall *below* random
  and/or majority. Do not hide this. It is the single most persuasive data
  point in the entire resubmission: it demonstrates concretely that scale
  does not guarantee routed-subset competence, that a "just use a bigger
  model" objection doesn't resolve the problem, and it is exactly the case
  the gate is built to catch. Feature it prominently rather than burying it.

**If a cell shows majority-class beating the verifier**, do not adjust the
narrative to avoid it. State it directly, and confirm (via the gate results
in §3.4) that the gate recommends fallback there. This is the single
strongest way to demonstrate the gate has real teeth rather than just
matching intuition.

### 3.4 Comment 4a — Gate infeasibility at zero-day inference

**What's needed:** Correct the framing rather than change the mechanism.

Add explicit language (in the gate's definition, moved to Section III per
§2.4, and repeated where the gate is evaluated in Results) stating:

> "The validation gate is a deployment-time admission-control check, not an
> inference-time mechanism. It is invoked once when a distribution shift is
> suspected (e.g., a new generator family enters the pipeline, or monitoring
> detects a change in disagreement rate or downstream error rate), using a
> small labeled calibration batch drawn from the disagreement subset of that
> new distribution. Once calibrated, the resulting escalate/fallback decision
> is applied to all subsequent unlabeled inputs in that distribution without
> requiring per-sample labels. This mirrors standard practice in ML ops for
> canary evaluation and model rollback and does not require ground truth at
> inference time for the inputs actually being classified in production."

This should fully resolve the "zero-day" framing objection — it was a
legitimate complaint about the paper's exposition, not about the mechanism
itself, which was already calibration-based.

### 3.5 Comment 4b — Sample-size sensitivity analysis

**What's needed:** A concrete, quantitative answer to "how many labeled
disagreement samples does the gate need?"

For every one of the 21 retrospective configurations already used for the
oracle gate validation, bootstrap-subsample the disagreement subset at
N = {10, 20, 50, 100, 200, and the full oracle N}, with ≥1,000 resamples per
(configuration, N) pair. At each draw, compute the gate's calibration
accuracies from only the subsampled labels and check whether the resulting
escalate/fallback decision matches the oracle (full-subset) decision.

Report:
- A plot: x-axis = N, y-axis = fraction of resamples (across all 21
  configs, or faceted by config) where the gate's decision matches the oracle
  decision. This is the headline sensitivity figure.
- A table: at each N, how many of the 21 configurations achieve ≥90% (or
  some stated threshold) decision-agreement rate with the oracle.
- An explicit practical recommendation, e.g., "N ≥ 50 labeled disagreement
  samples achieves oracle-agreement in X/21 configurations, providing a
  concrete minimum calibration-batch-size guideline."

Be honest if small N is noisy for close cases (e.g., the FLUX/SigLIP2/
GPT-5.5 near-tie already flagged in the paper) — that's expected and
consistent with the rest of the paper's tone (paired-bootstrap CIs are used
elsewhere already; this is the same rigor applied to gate reliability).

### 3.6 Comment 4c — Extend the gate to select among f1, f2, f3

**What's needed:** A genuine methodological extension, not just a reframe.

Generalize the binary gate to a **three-way selection rule**:
calibrate accuracy of f_artifact, f_semantic, and f_verifier on the same
labeled calibration sample from the disagreement subset, and route
disagreement-subset inputs to whichever of the three has the highest
calibration accuracy (with the existing binary gate as the special case
where f_verifier is compared only against max(f1, f2)).

This directly resolves the cross-family failure mode (Table 11), where
FFT-only beats the full cascade in 5/6 pairings — the three-way gate should
select FFT there instead of defaulting to "verifier or fallback-to-worse-
of-the-two-cheap-detectors." Validate this extension across the same 21
configurations plus the cross-family table's 6 configurations (27 total),
reporting decision accuracy the same way as the binary gate.

**This is the strongest available answer to R1's overall "limited practical
benefit" critique**, because it turns the gate from "a safety check that
sometimes says don't bother" into "a router that always makes the best
available choice among all three detectors, provably matching-or-beating
every individual detector in every tested condition." That is a much
stronger and more defensible claim than anything in the current draft.

---

## 4. Experiment Execution Order (recommended)

1. **Random + majority-class baselines** (§3.3) — cheapest, purely
   post-hoc on existing predictions, and determines whether the reframe is
   even supportable. Do this before anything else.
2. **Best-cheap-detector baseline** alongside the above — same pass.
3. **Three-way gate** (§3.6) — also computable entirely from existing stored
   predictions (f1, f2, f3 outputs already exist for every configuration).
   No new model inference required.
4. **Gate sensitivity sweep** (§3.5) — bootstrap subsampling over existing
   labeled data, no new inference required.
5. **Expanded in-domain evaluation sets** (§3.1) — the only step requiring
   new model inference (more Qwen/GPT-5.5 calls). Do this last so the
   reframe and gate-validation numbers are locked in first; if 3.1 shifts
   any headline number, only that section needs re-checking.

Steps 1-4 use data you already have on disk. Step 5 is the only one that
requires running detectors on new samples.

---

## 5. Section-by-Section Edit Map

| Section | Change |
|---|---|
| Title | Reframe toward the gate (§2.1) |
| Abstract | Rewrite with gate as headline (§2.2) |
| I. Introduction / Contributions | Reorder C-list, gate becomes C1 (§2.3) |
| III.D | Introduce gate definition immediately after Eq. 7 (§2.4) |
| V. Results | Add "roadmap of results" paragraph up front (§3.2); regenerate Tables 2/5/6/7 at larger N (§3.1); add baseline columns to Tables verifier_comparison, e8_results, biggan_shift, cross_family_deployment (§3.3) |
| V (new subsection) | Random/majority/best-cheap baseline summary table (§3.3) |
| V (new subsection) | Gate sensitivity analysis: figure + table (§3.5) |
| V (new subsection) | Three-way gate validation across 27 configs (§3.6) |
| VI. Discussion | Reframe every negative result as "expected under Case X, and correctly caught by the gate" rather than standalone limitation (§2.5) |
| VI.I (moves earlier structurally, but content stays, expanded) | Add zero-day/deployment-time clarification (§3.4); add sensitivity results; add three-way extension |
| Limitations | Add honest note on GenBuster N ceiling if applicable (§3.1) |
| Conclusion | Rewrite around: disagreement identifies difficulty (RQ1) → whether to act on it is conditional and formalized (RQ2/Eq.7) → the validation gate operationalizes this condition, is empirically accurate (20-21/21 → confirm with expanded N), degrades gracefully with small calibration samples, and extends to full three-way routing that matches-or-beats every individual detector across all tested conditions (RQ3) |

---

## 6. Response to Reviewers — Strategy Notes

For each of R1's four comments, the response document should have three
parts per the required template (concern / response / action). Suggested
substance for the "response" portions:

- **Comment 1:** Acknowledge the boundary-metric gap directly, point to the
  gate (now Section III, not VI) as the a priori mechanism, clarify it
  requires a small calibration sample rather than being fully zero-shot,
  and report the expanded evaluation sets with updated CIs.
- **Comment 2:** Acknowledge the apparent contradiction plainly, then
  explain the reframe: the varying outcomes across settings are not
  presented as evidence the method works, but as the evidence base
  motivating why a validation gate is necessary, and the gate is shown to
  correctly navigate all of them. Point to the new roadmap paragraph and the
  three-way gate result as the direct resolution.
- **Comment 3:** Report the random/majority/best-cheap baseline results in
  full, including any cells where the verifier does not clear them, and
  point to the gate correctly recommending fallback in those cells as
  evidence the framework accounts for exactly this failure mode.
- **Comment 4:** Separately address (a) the zero-day/deployment-time
  clarification, (b) the sensitivity sweep with a concrete N
  recommendation, and (c) the three-way extension, treating this as three
  sub-responses since the reviewer raised three distinct sub-points.

Do not get defensive anywhere in this document. R1's reading of the original
draft was reasonable given how it was written. The response should read as
"you correctly identified that the paper's framing did not match its actual
contribution; here is the corrected framing and the new evidence that
supports it," not "you misunderstood the paper."

---

## 7. Risk Assessment

- **Primary risk:** the random/majority-class baseline check (§3.3, do
  first) reveals that a verifier's headline result doesn't clearly clear a
  naive baseline in a case currently presented as a positive result (e.g.,
  if FLUX/GPT-5.5's +0.4% doesn't hold up against majority-class). If this
  happens, do not suppress it — this doesn't kill the paper, since the
  paper's contribution has been reframed to be about the gate's ability to
  correctly identify *both* helpful and unhelpful escalation conditions.
  A result where the gate correctly demotes a previously-reported "positive"
  case to fallback is a *stronger* demonstration of the gate's value than
  the original claimed win.
- **Secondary risk:** three-way gate (§3.6) doesn't cleanly resolve the
  cross-family case if f1/f2/f3 calibration accuracies are noisy at small
  N. Mitigate by reporting this transparently as part of the sensitivity
  analysis rather than only reporting the oracle (full-N) three-way result.
- **Reviewer assignment risk:** the cover letter explicitly warns reviewers
  may change. If a new R1-equivalent reviewer is assigned, the reframed
  paper needs to stand on its own without relying on "the previous reviewer
  said X" context — the response-to-reviewers document is for the editor
  and original reviewers, but the manuscript itself must independently read
  as coherent and well-supported.

---

## 8. Definition of Done

Before resubmitting, confirm all of the following are true:

- [ ] Title and abstract lead with the validation gate, not the cascade
- [ ] Contributions list has the gate as C1
- [ ] Gate definition appears in Section III, immediately tied to Eq. 7
- [ ] A "roadmap of results" paragraph precedes the detailed negative
      results in Section V, framing them as expected/diagnostic
- [ ] Every verifier-on-disagreement-subset result has random, majority-
      class, and best-cheap-detector baselines reported alongside it
- [ ] Any case where a baseline beats the verifier is reported honestly and
      cross-referenced to the gate correctly catching it
- [ ] In-domain evaluation sets expanded to ≥2,000 samples where source data
      allows; all downstream tables regenerated
- [ ] Gate reframed explicitly as a deployment-time, calibration-sample-based
      check, not a zero-day/label-free inference mechanism
- [ ] Sample-size sensitivity sweep (N = 10/20/50/100/200/oracle) completed
      across all 21 (+6 cross-family = 27) configurations, with a concrete
      minimum-N recommendation
- [ ] Three-way gate (f1/f2/f3 selection) implemented and validated across
      all 27 configurations, explicitly resolving the cross-family failure
      mode
- [ ] Discussion section reframes every negative result as gate-relevant
      evidence rather than standalone limitation
- [ ] Response-to-reviewers document drafted per §6, non-defensive in tone
- [ ] Full manuscript re-read once, end to end, as if by a first-time
      reader with no knowledge of the review history, to confirm the gate
      reads as the paper's thesis throughout