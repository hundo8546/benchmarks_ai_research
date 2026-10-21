# Response to Reviewers — Second Revision

We thank both reviewers for another careful, specific reading. Reviewer 1's ten
comments identified real issues in the previous revision, several of them genuine
bugs rather than exposition gaps — most significantly, a provenance bug in which the
Limitations-section class-wise numbers and Table 5's cascade accuracy had been
computed against two different trained-probe snapshots saved roughly two hours
apart, and an incomplete-GPT-5.5-coverage issue in the same table that silently fell
back to the semantic probe's prediction wherever a verifier answer was missing. We
have rebuilt the entire experimental pipeline around two canonical scripts
(`routing/run_pipeline.py` for data generation, `routing/generate_paper_results.py`
for every reported table/figure, both released alongside the code), fixed every bug
identified below, and re-run approximately 18,500 new GPT-5.5 evaluations (via the
OpenAI Batch API) to close the remaining gaps that required new data rather than
re-analysis of existing data. We address each comment in turn.

---

## Reviewer 1

### Comment 1 — Independent deployment-set evaluation of the validation gate

**Concern:** Section VI.I's gate validation evaluates bootstrap agreement against an
oracle computed from the same disagreement pool used for calibration, establishing
resampling stability rather than generalization to unseen inputs. The N≥100
recommendation is not adequately established (11/35 configurations miss the
criterion at that size).

**Response:** We agree, and have added a fully independent evaluation
(Section VI.I.1, "Independent (Held-Out) Gate Evaluation," new
Table "gate_holdout"). For each of the 35 configurations we now repeatedly
(500 draws) split the disagreement subset into disjoint calibration and test
halves, freeze the gate's decision on calibration only, and apply it unchanged to
the test half, reporting test-set accuracy, regret relative to the best fixed
detector chosen with test-half oracle knowledge, and the harmful-escalation rate.
Three configurations have disagreement pools under 40 samples and cannot support a
meaningful split; we report the remaining 32. Under this genuinely held-out
protocol, our significance-aware rule (Comment 2) reaches ≥90% test-set agreement
in 18/32 at N=50, 21/32 at N=100, and 24/32 at N=200 — visibly lower than the
same-pool-oracle figures (21/35, 24/35, 28/35). We have revised our practical
recommendation accordingly: N≥100 is reliable in roughly two-thirds of eligible
configurations under genuine held-out evaluation, not the majority previously
implied; we now recommend N≈200 or the full available pool, with an explicit
caveat that close-margin configurations remain unreliable even then.

**Action:** New Section VI.I.1 and Table "gate_holdout." Revised sample-size
recommendation throughout. Old same-pool sensitivity table retained alongside
(retitled "Same-Pool-Oracle Sensitivity") since it answers a genuinely different
question (resampling stability) that remains useful, but is now clearly
subordinate to the held-out result.

---

### Comment 2 — Statistical significance of the escalation decision

**Concern:** The gate enables escalation whenever the verifier's empirical accuracy
exceeds both cheap detectors by any positive margin, which can be sampling noise.
Calibration restricted to the disagreement subset also cannot establish agreement-path
reliability.

**Response:** We replaced the point-estimate rule with a prespecified,
uncertainty-aware one: escalate only if the paired-bootstrap 95% confidence
interval (10,000 resamples) on the accuracy gap lies entirely above zero. Evaluated
under the same held-out protocol as Comment 1, this reduces the mean
harmful-escalation rate by roughly an order of magnitude relative to the naive rule
at every tested N (e.g. 6.7% → 0.3% at N=50), at a modest cost in mean regret. We
also added an agreement-path audit across every evaluated dataset (not only
GenBuster): in-domain agreement-path accuracy is 96.8–99.8% with 0.2–3.2%
correlated-error rates; under SDXL/FLUX deployment shift it drops to 83.9–91.8%
with correlated-error rates 4–8× higher (8.2–16.1%) — errors disagreement-based
routing structurally cannot catch, now quantified rather than asserted.

**Action:** Significance-aware decision rule specified in pseudocode
(Section VI.I). New agreement-path audit results folded into Section VI.I and the
Limitations discussion.

---

### Comment 3 — Apparent equivalence of the binary gate and three-way selector

**Concern:** The binary gate (escalate iff the verifier beats both cheap detectors,
else fall back to the stronger of the two) already selects the best of all three
detectors, conflicting with the later claim that it "cannot represent
artifact-detector selection." The differing results at N=10 need explanation.

**Response:** The reviewer is correct, and we retract the "cannot represent" claim.
We now give exact pseudocode for both rules (Section VI.I) and state plainly that,
as decision procedures, they select the identical action in every case: the binary
gate's fallback branch is, by its own stated definition, `argmax(A1, A2)`. What
differed was not the gate's behavior but what our original sensitivity analysis
measured — a Boolean escalate/fallback agreement check for the binary rule versus a
full three-valued identity check for the three-way rule. We traced the one place
this produces different reported numbers (N=10: 13/35 vs. 12/35) to a single
configuration where resampled-accuracy ties between the two cheap detectors are
common enough at that coarse sample size (a binomial tie probability of ≈3.5%,
confirmed exactly against the observed tie count) that `argmax`'s first-index
tie-break affects the three-way check but not the binary one; the arithmetic closes
exactly (84 shared mismatches + 23 tie-driven mismatches = 107/1,000, matching the
observed gap), and the discrepancy vanishes by N=20.

**Action:** Exact pseudocode for both rules added to Section VI.I. The "cannot
represent" claim corrected to describe the three-way rule's actual contribution
(reporting detector identity explicitly, not a different decision). Full
mechanical explanation of the N=10 divergence added.

---

### Comment 4 — Real-image substitution in the deployment benchmark

**Concern:** Replacing AIGIBench's bundled PASCAL VOC real images after observing
GPT-5.5 misclassify 82–97% of them is a selection-after-failure concern. Results on
the original benchmark, and on an independently chosen real-image source, should be
reported.

**Response:** We now report both (new Table "real_image_sources," Section IV.B).
Under the original PASCAL VOC images, the full cascade — not only GPT-5.5 — degrades
severely: accuracy collapses to 72.9–77.9%, driven by an agreement-path accuracy of
only 43–53%. The local FFT artifact detector alone misclassifies 83–85% of these
photographs as fake, essentially matching GPT-5.5's own misclassification rate
despite using an entirely different, non-VLM signal. This shows the substitution
was not selected to flatter GPT-5.5 specifically: the original benchmark degrades
every tested detector, consistent with a genuine confound in the archival,
heavily-compressed source images rather than a VLM-specific calibration issue. We
additionally evaluated an independently chosen source, COCO val2017 (Flickr-sourced,
CC-licensed, sharing no images with PASCAL VOC or the ImageNet pool used elsewhere,
with distinct compression and resolution characteristics), on which cascade accuracy
is 89.6–94.1%, closely tracking the ImageNet-sourced benchmark used throughout this
paper. We have narrowed the recommendation accordingly: we do not suggest
categorically replacing genuine compressed photographs in VLM forensic evaluations;
we report this specific benchmark's pre-existing confound and provide the original,
substituted, and independent results side by side so readers can judge for
themselves.

**Action:** New Table "real_image_sources" (Section IV.B) with original-PASCAL-VOC,
COCO, and ImageNet-substituted results for all SDXL/FLUX backbone cells. Narrowed
recommendation language. ~9,766 new GPT-5.5 calls (Batch API, 100% valid) to
produce this table.

---

### Comment 5 — Reconciling class-wise results with Table 5

**Concern:** The reported SDXL real/fake recall figures and Table 5's overall
cascade accuracy do not arithmetically reconcile.

**Response:** This pointed us to a genuine bug, not an arithmetic slip in the text:
the two sets of numbers had been computed against different trained-probe snapshots
(the SD1.4-trained stale probes were retrained roughly two hours apart between two
separate runs, and the Limitations-section class-wise figures and Table 5's cascade
accuracy each used one of the two without either set being aware of the other). We
rebuilt both from a single, current, reproducible snapshot, with complete (100%)
GPT-5.5 coverage of the disagreement subset via the OpenAI Batch API (previously
partial, which is why some disagreement rows had silently fallen back to the
semantic probe's own prediction rather than a verifier answer). Under this
corrected, internally consistent accounting, both real-class and fake-class recall
increase under escalation in all four backbone/dataset cells — a stronger and
simpler result than, and a correction of, the earlier claim that real-class recall
declined. Full TP/FP/TN/FN confusion matrices for all four cells are released
alongside the per-sample prediction files.

**Action:** Table 5 and the Limitations-section class-wise paragraph rebuilt from
one consistent, 100%-coverage probe snapshot. Confusion matrices released.
Provenance of the original discrepancy documented in the Limitations section.

---

### Comment 6 — Dataset splitting procedure

**Concern:** The stated 50%/20%/40% nested-split formula does not reproduce the
paper's own reported evaluation-set sizes (1,000 GenBuster, 2,000 each for
SD1.4/BigGAN). The procedure needs full documentation, exact partition counts, and
duplicate checks for real images reused across benchmarks.

**Response:** The previous description was aspirational rather than an accurate
description of the code, and we have rewritten Section V.A to match what the
pipeline actually does: a single stratified evaluation-set holdout at a fixed
target size per dataset (1,000 for GenBuster — its full pool is only 2,000 samples,
disclosed as a ceiling; 2,000 each for SD1.4/BigGAN, drawn from 12,000-sample
pools), with every detector trained on the complementary split (the full pool minus
the evaluation set). No further nested validation holdout is used; a fixed-hyperparameter
logistic-regression classifier does not require one. We also found and fixed one
concrete consequence of the previous ambiguity: the CLIP probe for GenBuster had
never been retrained after the evaluation set was later expanded, so roughly 40% of
its "held-out" evaluation set had in fact been part of its own training data; it is
now retrained cleanly under the documented split. We additionally checked, by exact
path comparison, for real-image reuse across benchmarks: 100% of SDXL's and FLUX's
real-class images are drawn from SD1.4's real pool, and 31.5%/33.1% of SD1.4's own
evaluation-set real images are also present, unmodified, in the SDXL/FLUX real
class — disclosed explicitly in the new Section V.A, distinct from (and not
affecting) any individual benchmark's own train/evaluation separation. Split
identifiers and the full overlap-check output are released alongside the code.

**Action:** Section V.A rewritten with exact, correct partition counts per dataset.
CLIP probe retrained leak-free. Real-image overlap across benchmarks measured and
disclosed (new paragraph, Section V.A). Split identifiers released.

---

### Comment 7 — Auditable account of the corrected verifier experiments

**Concern:** Exact model identifier, API configuration, reasoning settings, token
limits (old and corrected), parsing rules, invalid-response handling, retry policy,
evaluation dates, and failed/truncated-call counts should be documented; the four
favorable SDXL/FLUX configurations need sample counts, class composition, routed
accuracies, paired CIs, and API-coverage completeness.

**Response:** New Section IV.C.1 ("GPT-5.5 and Qwen Verifier Audit") provides this
in full. Model identifier (`gpt-5.5` via the OpenAI Responses API); no explicit
reasoning/verbosity/temperature parameter is set, disclosed as relying on API
defaults rather than implying a configured value we do not control. The original
output-token cap (16–100 tokens) is documented alongside the corrected escalating
budget (2,048→4,096 tokens across up to 5 retries); because gpt-5.5 is a reasoning
model that spends output-token budget on a hidden reasoning pass, the low original
cap frequently produced empty or mid-word-truncated responses, which earlier code
silently defaulted to a fixed label. The corrected parser accepts only an exact
"REAL" or "FAKE" response and never guesses. We identified and corrected 19
previously-mislabeled truncated responses in the SDXL/FLUX disagreement predictions
that had escaped the original truncation-detection logic. We do not have per-call
timestamps for predictions made before this correction (only file-modification-time
proxies, clustering within a roughly 9-hour window); every call made as part of
this correction carries an explicit UTC timestamp, retry count, and full raw
response text, released with the per-sample files. GPT-5.5 coverage is complete
(100%) for the four SDXL/FLUX configurations and the two new real-image-source
benchmarks (Comment 4), achieved via ~18,500 new calls over the OpenAI Batch API;
coverage remains partial and explicitly disclosed for the Qwen columns of Table 5
and the previously-noted GenBuster/SD1.4 cells of the verifier-comparison table.

**Action:** New Section IV.C.1 with full audit detail. Per-sample prediction files
with timestamp/retry/token columns released. 19 previously-mislabeled rows
corrected.

---

### Comment 8 — Strengthen the routing comparisons

**Concern:** The entropy baseline uses artifact-only confidence at a fixed
threshold, which does not establish superiority over confidence routing generally.
Table 11's 0.917 vs. 0.898 needs reconciliation with the "no measurable
improvement" claim. The contextual bandit needs an explicit reward, exploration
procedure, and training algorithm, or should not be called a bandit.

**Response:** We built the joint artifact-and-semantic confidence-routing baseline
requested. It escalates when the less confident of the two cheap probes has entropy
above a threshold. The threshold is fit on a separate validation split to target the
disagreement rule's escalation rate. Non-escalated inputs keep the semantic probe's
prediction, and escalated inputs get the same verifier labels as the disagreement rule
(on SDXL/FLUX, the uniform GPT-5.5 run that labels every image). We evaluated it on the
three in-domain benchmarks and on the four SDXL/FLUX configurations behind our positive
result (new Table `confidence_routing`, Section V).

The reviewer's concern is partly borne out. In-domain, disagreement is significantly
more accurate on SD 1.4 (−2.2 points for the joint rule) and BigGAN (−0.7) and tied on
GenBuster. Under the SDXL/FLUX shift, disagreement has no accuracy advantage. The joint
rule is significantly better on SDXL/SigLIP2 (+1.4 points, 95% CI [+0.6, +2.1]) and tied
on the other three, and on FLUX it matches the disagreement rule while escalating about
10 points fewer inputs. With escalation counts matched exactly on the evaluation set,
the joint rule is significantly better in three of the four shifted configurations
(+0.9 to +1.7 points). We have revised C4 and the RQ2 conclusion accordingly.
Disagreement is a calibration-free routing signal that stays close to a calibrated
confidence rule (never more than 1.7 points behind). It is not a more accurate signal
in general.

On Table 11: the previously reported
0.917 figure is not reproducible from any current per-sample data (the underlying
protocol has since been superseded); recomputed from the current, single,
consistent pipeline, the learned classifier and the disagreement rule produce
byte-identical predictions (0.899 vs. 0.899, paired-bootstrap CI on the difference:
[0.000, 0.000]). A previously used alternative protocol, reproduced on current data
as a robustness check, gives a small difference in the opposite direction (0.900
vs. 0.903) that is also not distinguishable from zero. Both protocols now support
"no measurable improvement" with an explicit significance test rather than a bare
point comparison. On the bandit framing: the reviewer is correct that a logistic
regression predicting an escalation decision, with no reward signal, no exploration
policy, and no reinforcement-learning training procedure, does not establish a
bandit was implemented. We have relabeled this baseline throughout as a "learned
escalation classifier" rather than implement an unneeded reinforcement-learning
variant of a method whose main empirical finding (it converges to the disagreement
rule) does not depend on that framing.

**Action:** New joint confidence-routing baseline, method paragraph and Table
`confidence_routing` (Section V, Routing Analysis), with paired-bootstrap CIs and an
exact-rate-matched robustness check; C4 and the RQ2 conclusion revised to match.
Table 11 numbers corrected and given a significance test under two protocols.
"Contextual bandit" relabeled "learned escalation classifier" throughout the paper.

---

### Comment 9 — Dataset shortcuts and reproducibility

**Concern:** BigGAN's near-perfect FFT performance warrants controls for
compression, resolution, and source differences; FFT/DCT feature definitions,
preprocessing, and training details should be fully specified.

**Response:** We directly tested and confirmed this concern for BigGAN. A
classifier using only an image's native width and height — no pixel content —
reaches 99.55% accuracy on BigGAN, within 0.15 points of the trained FFT detector's
99.7%; forcing the real class through the same native-resolution downsampling the
fake class structurally undergoes before the shared feature pipeline collapses the
same trained detector to 50.1%, exact chance. We extended the compression-robustness
check to the FFT/SigLIP2/DINOv2 detectors (previously CNNSpot/CLIP only), on the
current (not stale) evaluation splits, and added two further common post-processing
operations — Gaussian blur and downscale-then-upscale — at multiple severities. The
detector's BigGAN accuracy is nearly invariant to JPEG compression (99.7%→99.65% at
quality 50) but collapses sharply, including below chance, under blur and
aggressive resizing (99.7%→29.2% at blur σ=2; 99.7%→33.9% at a 4× downscale-upscale
round-trip) — a pattern consistent with a detector keyed to a coarse, resize-related
signature rather than fine generative texture. SD1.4 shows a partial version of the
same confound; GenBuster, where both classes share identical native resolution,
shows none, serving as a clean negative control. The exact FFT/DCT feature
definition (32-bin radial FFT magnitude profile, L2-normalized; 8×8-block DCT with
the bottom-right 3×3 high-frequency submatrix pooled into 8 summary statistics;
3 color-channel coherence correlations; 256×256 preprocessing resize;
`LogisticRegression(C=0.1, random_state=42)`) is now documented in full.

**Action:** Resolution-only control classifier and resolution-matching ablation
(new figure and Limitations discussion). Compression robustness extended to all
three detectors; two new post-processing perturbations added. Full FFT/DCT
reproducibility documentation added. This finding is flagged as a direct caveat on
the in-domain BigGAN artifact-detector baseline used in the cross-family deployment
comparison (Section VI.B).

---

### Comment 10 — Complete the cost analysis

**Concern:** Token-cost reporting is promised but not substantiated in the main
tables. The positive deployment settings should compare the cascade directly
against standalone cheap detectors and uniform GPT-5.5 inference, not only report a
reduced invocation count.

**Response:** New Section VI.J ("Cost Accounting and Cascade vs. Uniform
Escalation") reports real, measured per-call token counts and verified gpt-5.5
pricing (Standard and Batch tiers, cited from current published pricing), clearly
separated from the hardcoded cheap-detector latency constants used elsewhere. We
closed the cascade-vs-uniform comparison gap directly: using the OpenAI Batch API,
we ran GPT-5.5 on every image in both SDXL and FLUX (8,000 calls, 100% successful),
not only the disagreement subset. Uniform escalation reaches 99.6%/97.1% accuracy at
$13.61/$18.47 for the full benchmark; the cascade reaches 89.3–94.0% at 58–64% of
that cost. We report this as an explicit, incomplete-but-substantial tradeoff —
3.0–7.8 percentage points remain achievable by uniform escalation that the cascade
does not recover — rather than implying the cascade matches uniform accuracy at
lower cost.

**Action:** New Section VI.J with a cost-accuracy table and figure directly
comparing cheap-only, cascade, and uniform GPT-5.5 policies on SDXL/FLUX.

---

### Closing paragraph (disagreement identifying difficulty; video framing)

We agree disagreement does not universally identify "difficulty" in a
detector-agnostic sense; Section V.B now states explicitly that on GenBuster the
semantic probe remains 96.1% accurate on disagreement cases, so the signal there
flags artifact-detector-specific failure, not a case both detectors find hard
(this also responds to Reviewer 2's Comment 2 below). The Limitations section now
states plainly that this study evaluates frame-level forensic detection and does
not validate temporal video forensics, and that claim should not be read more
broadly than that single-frame scope.

---

## Reviewer 2

### Comment 1 — Footnote the bootstrap-independence clarification on the sensitivity table

**Action:** The clarification ("for configurations whose disagreement subset
contains fewer than 200 unique samples, the N=100/N=200 columns are bootstrap
resamples... not a stand-in for independent human-labeled examples") is now
included directly in the sensitivity table's caption, in addition to the body text.

### Comment 2 — Rephrase "routing identifies difficult inputs, but the selected verifier cannot improve them"

**Action:** Rewritten exactly along the lines suggested: on GenBuster, the routed
inputs are not intrinsically difficult for every cheap detector — the semantic
probe remains 96.1% accurate on the same disagreement subset. Disagreement there
primarily captures inputs where the artifact detector degrades while the semantic
detector remains reliable, and escalating to a weaker verifier replaces a correct
semantic prediction rather than resolving genuine difficulty.

### Comment 3 — Precision on the "significantly outperforms random triggering" claim

**Action:** Rewritten with the reviewer's own numbers: the gain is practically
substantial only on SD1.4 (0.867 vs. 0.839); on BigGAN the gain is statistically
outside the random baseline's CI but small in absolute terms (0.998 vs. 0.994,
consistent with a ceiling effect); on GenBuster the cascade does not outperform the
random baseline (0.898 vs. 0.904).

### Comment 4 — Unified Case 1/2/3/Breakdown summary table

**Action:** Added as a figure (Section VI.A, "The Three Cascade Cases in
Practice") rather than a text table, mapping every tested configuration in the
paper onto its Case assignment with the specific evidence justifying each.

---

## Summary of material changes

- Root-cause bugs found and fixed: CLIP-probe training leakage (GenBuster);
  inconsistent dataset-splitting procedure; a probe-snapshot-versioning bug behind
  Table 5/class-wise reconciliation; a GPT-5.5 truncation-detection gap (19 rows);
  incomplete GPT-5.5 coverage silently falling back to a weaker prediction.
- ~18,500 new GPT-5.5 evaluations (OpenAI Batch API) to achieve complete coverage
  for the four SDXL/FLUX configurations, the original-PASCAL-VOC benchmark, the new
  independent (COCO) real-image-source benchmark, and uniform/always-escalate
  evaluation of both SDXL and FLUX.
- New analyses: held-out gate validation, significance-aware escalation rule,
  agreement-path audit, joint confidence-routing baseline, resolution-shortcut
  control and ablation, extended compression/post-processing robustness, complete
  cost accounting with a direct cascade-vs-uniform comparison.
- Entire experimental pipeline consolidated into two canonical, released scripts
  (`routing/run_pipeline.py`, `routing/generate_paper_results.py`) replacing
  approximately 150 previously scattered scripts, so every reported number has a
  single, traceable, reproducible source.
