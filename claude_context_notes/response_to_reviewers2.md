# Response to Reviewers: Second Revision

We thank both reviewers for another careful, specific reading. Reviewer 1's ten
comments identified real issues in the previous revision, several of them genuine
bugs rather than exposition gaps. The most significant was a provenance bug in which the
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

### Comment 1: Independent deployment-set evaluation of the validation gate

**Concern:** Section VI.I's gate validation evaluates bootstrap agreement against an
oracle computed from the same disagreement pool used for calibration, establishing
resampling stability rather than generalization to unseen inputs. The N≥100
recommendation is not adequately established (11/35 configurations miss the
criterion at that size).

**Response:** We agree, and have added a fully independent evaluation
(Section VI.I.1, "Independent (Held-Out) Gate Evaluation," held-out rows of the
gate-reliability Table `gate`). For each of the 35 configurations and each calibration
size N ∈ {50, 100, 200}, we draw 250 random splits of the disagreement subset into
disjoint calibration and test sets of N samples each, plus 250 half/half splits of
the full pool. We freeze the gate's decision on calibration only and apply it
unchanged to the test set. We report agreement with the test set's own oracle,
regret, the harmful-escalation rate, and the missed-beneficial-escalation rate
(verifier strictly best on test, but the frozen decision fell back). Where a pool
is too small for N, N is capped at half the pool: GenBuster/GPT-5.5 and
GenBuster/Qwen at 129 for N=200. The table caption lists both. Two
configurations (BigGAN/Qwen and BigGAN/GPT-5.5, n=14) have disagreement pools under
40 samples and cannot support a meaningful split; we report the remaining 33.
(SD1.4/GPT-5.5 is included because its GPT-5.5 coverage is now complete, growing its
pool from 23 to 532 samples; see Comment 7.) Under this held-out protocol, our
significance-aware rule (Comment 2) reaches ≥90% test-set agreement in 17/33 at
N=50, 21/33 at N=100, and 25/33 at N=200. That is visibly lower than the same-pool
figures (22/35, 23/35, 28/35). We have revised our practical recommendation
accordingly: N≥100 is reliable in only about two-thirds of eligible configurations,
so we now recommend N≈200 or the full available pool, and we state explicitly that
close-margin configurations can remain unreliable even then. For the four SDXL/FLUX
GPT-5.5 configurations behind our positive result, the table also reports how often
the significance-aware gate enables escalation. Three escalate in ≥85% of draws at
every N. FLUX/SigLIP2, the smallest-margin case, escalates in only 38% of draws at
N=50 and 63% at N=100 (91% at N=200); the table reports this and the text notes it.

**Action:** New Section VI.I.1, and a single gate-reliability Table `gate`. Its
held-out rows report the capped-N footnote, missed-beneficial-escalation rates and
positive-cell escalation rates. Revised sample-size recommendation throughout. The
earlier same-pool results are kept as the first rows of the same table, labelled as
measuring resampling stability, since that remains a useful but secondary question.

---

### Comment 2: Statistical significance of the escalation decision

**Concern:** The gate enables escalation whenever the verifier's empirical accuracy
exceeds both cheap detectors by any positive margin, which can be sampling noise.
Calibration restricted to the disagreement subset also cannot establish agreement-path
reliability.

**Response:** We replaced the point-estimate rule with a prespecified,
uncertainty-aware one: escalate only if the paired-bootstrap 95% confidence
interval (10,000 resamples) on the accuracy gap lies entirely above zero. Evaluated
under the same held-out protocol as Comment 1, this reduces the mean
harmful-escalation rate by roughly an order of magnitude relative to the naive rule
at every tested N (e.g. 7.3% → 0.3% at N=50). The cost is that it escalates less
often. It misses a beneficial escalation in 14–18% of draws (naive rule: 2–9%), and
its mean regret is higher at every N. As a result it matches the test oracle less
often than the naive rule once calibration data is plentiful (26 vs. 29 of 33 with
the half pool). The paper now presents the two rules as a
risk trade-off rather than one dominating the other. We
also added an agreement-path audit across every evaluated dataset (not only
GenBuster): in-domain agreement-path accuracy is 96.8–99.8% with 0.2–3.2%
correlated-error rates; under SDXL/FLUX deployment shift it drops to 83.9–91.8%
with correlated-error rates 4–8× higher (8.2–16.1%). These are errors disagreement-based
routing structurally cannot catch, now quantified rather than asserted.

**Action:** Significance-aware decision rule specified in pseudocode
(Section VI.I). New agreement-path audit results folded into Section VI.I and the
Limitations discussion.

---

### Comment 3: Apparent equivalence of the binary gate and three-way selector

**Concern:** The binary gate (escalate iff the verifier beats both cheap detectors,
else fall back to the stronger of the two) already selects the best of all three
detectors, conflicting with the later claim that it "cannot represent
artifact-detector selection." The differing results at N=10 need explanation.

**Response:** The reviewer is correct, and we retract the "cannot represent" claim.
We now give exact pseudocode for both rules (Section VI.I) and state plainly that,
as decision procedures, they select the identical action in every case: the binary
gate's fallback branch is, by its own stated definition, `argmax(A1, A2)`. What
differed was not the gate's behavior but what our original sensitivity analysis
measured: a Boolean escalate/fallback agreement check for the binary rule versus a
full three-valued identity check for the three-way rule. We traced the one place
this produces different reported numbers (N=10: 13/35 vs. 11/35) to two
configurations (SD1.4/SigLIP2 at 100% training fraction, and FLUX/DINOv2, both with
the Qwen-7B verifier). At that coarse sample size the resampled accuracy of the weaker
cheap detector ties the stronger one's (19 and 25 of 1,000 resamples) or exceeds it
(10 and 13). This changes which cheap detector is selected, which the three-way check
counts and the binary check does not. The arithmetic closes exactly for both: for
SD1.4, 93 shared mismatches plus 18 wrong-cheap-detector mismatches equals 111/1,000,
matching the observed 90.7% vs. 88.9% gap; for FLUX/DINOv2, 99 plus 19 equals
118/1,000, matching 90.1% vs. 88.2%. (The previous version reported 13/35 vs. 12/35,
a one-configuration gap; the counts moved because several configurations' data changed
in this revision, such as completed GPT-5.5 coverage, and the resampling stream is shared
across configurations.) The discrepancy vanishes by N=20.

**Action:** Exact pseudocode for both rules added to Section VI.I. The "cannot
represent" claim corrected to describe the three-way rule's actual contribution
(reporting detector identity explicitly, not a different decision). The paper
explains the mechanism of the N=10 divergence; the full count-level accounting is
given above.

---

### Comment 4: Real-image substitution in the deployment benchmark

**Concern:** Replacing AIGIBench's bundled PASCAL VOC real images after observing
GPT-5.5 misclassify 82–97% of them is a selection-after-failure concern. Results on
the original benchmark, and on an independently chosen real-image source, should be
reported.

**Response:** We now report both (new Table "real_image_sources," Section IV.B).
Under the original PASCAL VOC images, the full cascade, not only GPT-5.5, degrades
severely: accuracy collapses to 72.9–77.9%, driven by an agreement-path accuracy of
only 43–53%. The local FFT artifact detector alone misclassifies 83–85% of these
photographs as fake, essentially matching GPT-5.5's own misclassification rate
despite using an entirely different, non-VLM signal. This shows the substitution
was not selected to flatter GPT-5.5 specifically: the original benchmark degrades
every tested detector, consistent with a genuine confound in the source images rather
than a VLM-specific calibration issue. The confound is concrete: the bundled PASCAL VOC
images are uniformly 256×256 while every fake is 1,024×1,024, so a classifier given only
image width and height separates the two classes perfectly. The COCO source is separable
by size in the same way, which is one reason we added the size-matched control described
under Comment 9. We
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

### Comment 5: Reconciling class-wise results with Table 5

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
increase under escalation in all four backbone/dataset cells. This is a stronger and
simpler result than, and a correction of, the earlier claim that real-class recall
declined. Full TP/FP/TN/FN confusion matrices for all four cells are released
alongside the per-sample prediction files.

Prompted by this comment, we audited every table and inline number in the paper against
the results pipeline and found the same class of problem elsewhere. Several tables
still carried values from earlier pipeline states:
- the GenBuster CLIP row of the main results (it still showed the pre-fix, leaky probe);
- the cheap-trigger, transfer, probe-degradation and cross-family tables;
- a per-generator table computed on an older 400-sample subset.

Every table and figure is now produced by one entry point
(`routing/generate_paper_results.py`) from the same canonical evaluation sets as the
main results table. Wherever two tables report the same quantity they now agree
exactly. For example, the cross-family FFT column equals the in-domain FFT accuracy,
and the 100% row of the probe-degradation sweep equals the main results. One
conclusion changed in the process, and the text says so: the BigGAN→GenBuster
cross-family pairing, previously +0.003 over in-domain FFT and not significant, is now
a small, significant gain (+0.030, 95% CI [+0.005, +0.055]).

**Action:** Table 5 and the Limitations-section class-wise paragraph rebuilt from
one consistent, 100%-coverage probe snapshot. Confusion matrices released.
The provenance of the original discrepancy is explained in this response (the
manuscript reports only the corrected figures). All
tables regenerated from a single pipeline entry point on the canonical evaluation
sets.

---

### Comment 6: Dataset splitting procedure

**Concern:** The stated 50%/20%/40% nested-split formula does not reproduce the
paper's own reported evaluation-set sizes (1,000 GenBuster, 2,000 each for
SD1.4/BigGAN). The procedure needs full documentation, exact partition counts, and
duplicate checks for real images reused across benchmarks.

**Response:** The previous description was aspirational rather than an accurate
description of the code, and we have rewritten Section V.A to match what the
pipeline actually does: a single stratified evaluation-set holdout at a fixed
target size per dataset (1,000 for GenBuster, whose full pool is only 2,000 samples,
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
class. This is disclosed explicitly in the new Section V.A, distinct from (and not
affecting) any individual benchmark's own train/evaluation separation. Split
identifiers and the full overlap-check output are released alongside the code.

**Action:** Section V.A rewritten with exact, correct partition counts per dataset.
CLIP probe retrained leak-free. Real-image overlap across benchmarks measured and
disclosed (new paragraph, Section V.A). Split identifiers released.

---

### Comment 7: Auditable account of the corrected verifier experiments

**Concern:** Exact model identifier, API configuration, reasoning settings, token
limits (old and corrected), parsing rules, invalid-response handling, retry policy,
evaluation dates, and failed/truncated-call counts should be documented; the four
favorable SDXL/FLUX configurations need sample counts, class composition, routed
accuracies, paired CIs, and API-coverage completeness.

**Response:** New Section IV.B.1 ("GPT-5.5 and Qwen Verifier Audit") provides this
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
benchmarks (Comment 4), achieved via ~18,500 new calls over the OpenAI Batch API.
Coverage is also now complete for the Qwen columns of Table 5 (the last 375
sample-backbone pairs were scored locally) and for the GenBuster/SD1.4 GPT-5.5 cells
of the verifier-comparison table (259 and 532 samples; the 716 missing labels had
already been returned by the corrected Batch run and are now merged). With full
coverage, GPT-5.5 on GenBuster is 0.907 vs. 0.961 for the best cheap detector (below
it), and on SD1.4 it is 0.814 vs. 0.784 (a +0.030 numerical edge whose paired 95% CI
[−0.019, +0.079] includes zero), so neither cell clears the baseline and no case
assignment changed.
For the four favorable SDXL/FLUX configurations, the second block of new Table
`verifier_subset` reports
the full disagreement-subset size (1,452–1,675), its real/fake composition (about
two-thirds fake), GPT-5.5 and best-cheap accuracy on the subset, and a
paired-bootstrap 95% CI on the gap (+8.1 to +20.4 points, every interval above zero),
all at 100% GPT-5.5 coverage. The baseline and gate analyses now also use these full
subsets with the uniform-run labels; previously they used only the samples that also
had Qwen coverage.

The size-matched control described under Comment 9 adds two further uniform GPT-5.5
runs over the same 7,315 SDXL/FLUX images, at 512 and 256 pixels, with the same model,
prompt, parser, output budget and Batch API settings (14,630 calls, all valid, run on
2026-10-22). Batch requests that failed for lack of account credit were resubmitted
unchanged, and 4 requests that stalled in the batch queue were completed through the
synchronous path. Section IV.B.1 documents this, and the per-sample files record which
path produced each answer.

**Action:** New Section IV.B.1 with full audit detail, and new Table
`positive_cells` (Section V). Per-sample prediction files with timestamp/retry/token
columns released. 19 previously-mislabeled rows corrected.

---

### Comment 8: Strengthen the routing comparisons

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
rule is significantly better on SDXL/SigLIP2 (+1.4 points, 95% CI [+0.6, +2.2]) and tied
on the other three, and on FLUX it matches the disagreement rule while escalating about
10 points fewer inputs. With escalation counts matched exactly on the evaluation set,
the joint rule is significantly better in three of the four shifted configurations
(+0.9 to +1.7 points).

This comparison broadened the paper's claim rather than only narrowing it. The
cascade's gain under deployment shift does not depend on the routing signal: with
GPT-5.5 as verifier, confidence routing improves on the stale semantic probe by 3.6–7.7
points across the four SDXL/FLUX configurations, against 3.4–7.9 for disagreement. The
contribution is therefore the gated cascade under deployment shift, and it applies with
either signal. We no longer claim disagreement is a more accurate signal than calibrated
confidence. We present it as the default because it needs no threshold, no probability
outputs and no tuning, and stays within 1.7 points of the calibrated rule. Disagreement
cannot say which detector is wrong on a single input, but with a labeled calibration
sample the disagreement subset shows which cheap detector is failing there. This is what
the gate's three-way selection uses, and the paper's Case 1/2/3 analysis relies on it.
C4, the confidence-routing discussion and the RQ2 conclusion are revised accordingly.

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

### Comment 9: Dataset shortcuts and reproducibility

**Concern:** BigGAN's near-perfect FFT performance warrants controls for
compression, resolution, and source differences; FFT/DCT feature definitions,
preprocessing, and training details should be fully specified.

**Response:** We directly tested and confirmed this concern for BigGAN. A
classifier using only an image's native width and height, with no pixel content,
reaches 99.55% accuracy on BigGAN, within 0.15 points of the trained FFT detector's
99.7%; forcing the real class through the same native-resolution downsampling the
fake class structurally undergoes before the shared feature pipeline collapses the
same trained detector to 50.1%, exact chance. We added a robustness check of the
FFT/SigLIP2/DINOv2 detectors on the current evaluation splits under JPEG compression,
Gaussian blur and downscale-then-upscale, each at multiple severities. The
detector's BigGAN accuracy is nearly invariant to JPEG compression (99.7%→99.65% at
quality 50) but collapses sharply, including below chance, under blur and
aggressive resizing (99.7%→29.2% at blur σ=2; 99.7%→33.9% at a 4× downscale-upscale
round-trip), a pattern consistent with a detector keyed to a coarse, resize-related
signature rather than fine generative texture. SD1.4 shows a partial version of the
same confound; GenBuster, where both classes share identical native resolution,
shows none, serving as a clean negative control. The exact FFT/DCT feature
definition (32-bin radial FFT magnitude profile, L2-normalized; 8×8-block DCT with
the bottom-right 3×3 high-frequency submatrix pooled into 8 summary statistics;
3 color-channel coherence correlations; 256×256 preprocessing resize; standardized
features with `LogisticRegression(C=0.1, max_iter=2000, random_state=42)`) is now
documented in full in the Implementation section, checked against the extractor code.

We also applied the same resolution control to the SDXL/FLUX deployment benchmark behind
the paper's positive result, which the reviewer's concern applies to just as much. Every
fake there is a 1,024×1,024 PNG while the real images are smaller JPEGs, and a
width-and-height-only classifier separates the classes with 98.9–100.0% accuracy under
all three real-image sources. We therefore built a size-matched copy of the benchmark in
which every real and fake image is center-cropped and resized to 256×256 PNG (both classes
downscaled), and repeated the evaluation with the same stale probes, nothing retrained,
and a new uniform GPT-5.5 run in which every image costs the same 119 input tokens (new subsection
"Size-Matched Control" in Section V, Table `sizematched`). The cascade's gain over the stale probe remains
significant in all four configurations (3.9–7.1 points), GPT-5.5 still beats the best
cheap detector on every disagreement subset, and disagreement routing still beats
escalating a random subset of the same size in three of the four, although by only
0.4–2.3 points, which the paper now states: most of the gain comes from the verifier's
strength on the escalated inputs rather than from which inputs disagreement selects. GPT-5.5 itself partly
used the size cue: its always-escalate accuracy falls by 2.6–3.3 points, mostly on real
photographs. A 512-pixel version of the control, which upscales almost every real image,
instead pushes the FFT probe below chance. The cascade's gain over the stale probe is
still significant in all four configurations at 512 pixels (3.4–7.6 points); only the
comparison with random routing depends on the size chosen. The paper reports this and
explains why 256 pixels is the appropriate control. Real images still carry their original JPEG
history, which we state as a limitation. The related-work discussion of dataset
shortcuts now cites the analysis the reviewer pointed to (Grommelt et al., ECCV
Workshops 2024).

**Action:** Resolution-only control classifier and resolution-matching ablation
(Limitations discussion). Size-matched control of the SDXL/FLUX deployment results (new
"Size-Matched Control" subsection and Table `sizematched`; C3, C6, abstract and Limitations updated). Compression robustness extended to all three detectors, and
two new post-processing perturbations added; the key results are in the Limitations
text and the full grid is in the released results file (`paper_results.xlsx`). Full FFT/DCT
reproducibility documentation added. This finding is flagged as a direct caveat on
the in-domain BigGAN artifact-detector baseline used in the cross-family deployment
comparison (Section VI.B).

---

### Comment 10: Complete the cost analysis

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
$13.61/$18.47 for the full benchmark. The cascade reaches 89.3–94.0% at 42–51% of
that cost. This is priced from the tokens of the images the cascade actually
escalates, which use more tokens per call than the average image, so the cost share
is higher than the 36–42% escalation rate. We report this as an explicit, incomplete but substantial tradeoff:
3.0–7.8 percentage points remain achievable by uniform escalation that the cascade
does not recover, and we do not imply that the cascade matches uniform accuracy at
lower cost. Under the size-matched control (Comment 9), where every image costs the same
number of tokens, the cascade costs 39–47% of uniform escalation.

Section VI.J also reports mean tokens per GPT-5.5 call for every dataset (e.g. 783
input / 96 output on SDXL, 783 / 177 on FLUX), with a citation for the pricing. The
formulation section now says that C_tok is reported wherever the verifier is an API
model, rather than for every method.

**Action:** New Section VI.J comparing cheap-only, cascade, and uniform GPT-5.5
policies on SDXL/FLUX (cost columns added to Table `e8_results`), plus per-dataset
token means.

---

### Closing paragraph (disagreement identifying difficulty; video framing)

We agree disagreement does not universally identify "difficulty" in a
detector-agnostic sense; Section V.B now states explicitly that on GenBuster the
semantic probe remains 96.1% accurate on disagreement cases, so the signal there
flags artifact-detector-specific failure, not a case both detectors find hard
(this also responds to Reviewer 2's Comment 2 below). The Limitations section and the
Conclusion now state plainly that this study evaluates isolated frames, i.e.
frame-level forensic detection, and does not validate temporal video forensics.

### Assessment questions (contribution)

We agree that disagreement routing, model cascades and selection with labeled calibration
data are established ideas, and that Eq. (4) is a direct decomposition rather than a
theoretical advance. The paper's contribution is empirical: when disagreement-based
escalation helps or harms AI-generated image detection, as the reviewer puts it. C2 now
says we evaluate, rather than propose, the routing rule, and the revisions above aim at
the quality of that evidence.

---

## Reviewer 2

### Comment 1: Footnote the bootstrap-independence clarification on the sensitivity table

**Action:** The clarification ("for configurations whose disagreement subset
contains fewer than 200 unique samples, the N=100/N=200 columns are bootstrap
resamples... not a stand-in for independent human-labeled examples") is now
included directly in the caption of the gate-reliability table (Table `gate`), in
addition to the body text.

### Comment 2: Rephrase "routing identifies difficult inputs, but the selected verifier cannot improve them"

**Action:** Rewritten exactly along the lines suggested: on GenBuster, the routed
inputs are not intrinsically difficult for every cheap detector, since the semantic
probe remains 96.1% accurate on the same disagreement subset. Disagreement there
primarily captures inputs where the artifact detector degrades while the semantic
detector remains reliable, and escalating to a weaker verifier replaces a correct
semantic prediction rather than resolving genuine difficulty.

### Comment 3: Precision on the "significantly outperforms random triggering" claim

**Action:** Rewritten to state the gain per dataset. Table `cheap_trigger` is now
recomputed on the same evaluation sets as Table `main_results`, and its random baseline
keeps the semantic prediction for inputs it does not escalate. The gain is practically
substantial only on SD1.4 (0.874 vs. 0.833). On BigGAN it is statistically outside the
random baseline's CI but small in absolute terms (0.995 vs. 0.990, consistent with a
ceiling effect). On GenBuster the cascade does not outperform the random baseline
(0.899 vs. 0.900).

### Comment 4: Unified Case 1/2/3/Breakdown summary table

**Action:** Added a compact table (Table `case_mapping`) in Section VI.B ("The Three
Cascade Cases in Practice") giving Case 1/2/3/Breakdown counts for each experimental
setting. The per-configuration assignment, with the evidence for each, is in the
released results file. Case 3 is assigned mechanically: the verifier must beat the best cheap
detector on the disagreement subset with a 95% CI above zero. Six configurations
qualify: the four SDXL/FLUX GPT-5.5 cells, BigGAN→GenBuster cross-family, and the
1% SigLIP2 probe-degradation cell.

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
  cost accounting with a direct cascade-vs-uniform comparison, and a size-matched
  control of the SDXL/FLUX deployment results (14,630 further GPT-5.5 calls).
- Entire experimental pipeline consolidated into two canonical, released scripts
  (`routing/run_pipeline.py`, `routing/generate_paper_results.py`) replacing
  approximately 150 previously scattered scripts, so every reported number has a
  single, traceable, reproducible source.
- Claims revised after correcting our own analyses:
  - Real-class recall on SDXL/FLUX increases under escalation, correcting the previous
    version's statement that it declined (R1 Comment 5).
  - BigGAN→GenBuster cross-family deployment is a small significant gain over
    in-domain FFT.
  - Under joint confidence routing (new), disagreement is not more accurate than a
    calibrated confidence rule under shift; the paper no longer implies it is
    (R1 Comment 8).
- BigGAN distant shift (Section V.D.7). The verifier accuracies are unchanged from the
  previous version (GPT-5.5 / Qwen-7B / Qwen-72B: 0.582 / 0.515 / 0.623). At native
  resolution the verifiers are strongly biased toward "real", and BigGAN fakes are
  natively 128x128, so we added a resolution test: resizing every image (real and fake)
  to a 512 px longest edge, with nothing else changed, raises GPT-5.5 to 0.710 (fake
  recall 0.175 to 0.435) and Qwen-7B to 0.761 (fake recall 0.045 to 0.700); Qwen-72B
  does not fit on our GPU and was not rerun. Because Qwen-7B then outranks GPT-5.5, the
  verifier ranking depends on input size, and we withdraw the previous version's
  statement that the 72B model performs best and that suitability cannot be inferred
  from scale; the paper now draws no conclusion about model scale. No verifier
  approaches stale SigLIP2 (0.907) in either condition, so the negative result
  (escalation hurts under this shift) holds in all runs. The BigGAN rows of Table
  `verifier_subset` report accuracy at both resolutions.
- Presentation: tables and figures consolidated. The paper keeps the previous
  version's five figures (redrawn from current data) and adds one, a bar chart of
  disagreement rates by dataset and semantic backbone that replaces the previous
  agreement-subset coverage table. It has 12 results tables, down from 14, despite the
  new analyses. Related tables are merged: verifier vs.
  best-cheap results in one table, SDXL/FLUX accuracy and cost in one, and gate
  reliability in one. Small tables (baseline counts, escalation breakdown,
  per-generator, robustness grid) became sentences in the text. Every number,
  including those summarized in prose, is released in `paper_results.xlsx`, which
  maps each value to the script and data that produce it.
