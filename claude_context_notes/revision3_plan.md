# Revision 3 Plan — close the gaps before resubmitting

A review of `paper_updates2.tex` against `response_to_reviewers2.md` and the pipeline
code found three kinds of problems:
1. The response letter promises content that the paper doesn't contain.
2. Some numbers disagree across tables, or one analysis is unfair to the baseline.
3. Stale text contradicts this round's new findings.

Very little new compute is needed. Most of the work is re-analysis of data already on
disk, followed by manuscript edits.

**Files in scope**
- Manuscript: `claude_context_notes/paper_updates2.tex`
- Response letter: `claude_context_notes/response_to_reviewers2.md`
- Results generator: `routing/generate_paper_results.py`
- Per-table code: `routing/pipeline/tables/*.py`
- Outputs: `routing/results/tab_*.csv`

---

## Ground rules

- **Run `git lfs pull` first.** Every CSV under `routing/results/` is an LFS object.
  Don't start any phase until `head routing/results/tab_main_results.csv` shows real
  CSV rows.
- **Never hand-edit a number in the `.tex`.** Every table cell and inline number comes
  from a `tab_*.csv` written by `generate_paper_results.py`. If a number has to change,
  change the code, regenerate, and then copy the value across.
- **No API or GPU spend without asking.** Phase 4 is the only part that needs it.
  Estimate the cost and call count, then wait for approval.
- **Keep the response letter in sync.** After each phase, update the matching comment
  in `response_to_reviewers2.md` so that every "Action:" line points to something that
  really exists in the paper, with the correct section number.
- **Commit after each phase,** with a message that names the reviewer comment it
  addresses.

---

## Phase 0 — Setup and baseline audit (no changes)

1. Run `git lfs pull`. Then run `python routing/generate_paper_results.py`, and record
   which tables regenerate cleanly and which fail.
2. Write `claude_context_notes/number_audit.md`. For every table and every inline
   number in `paper_updates2.tex`, record:
   - the value in the `.tex`
   - the value in the CSV
   - the source CSV and column
   - whether they match

   Known mismatches to confirm:

   | Quantity | Table `main_results` | Other tables |
   |---|---|---|
   | SD1.4 cascade (SigLIP2) | 0.874 | 0.867 in `cheap_trigger`, `transfer`, `limited_data` (100%) |
   | SD1.4 DINOv2, semantic only | 0.877 | 0.890 in `limited_data` (100%) |
   | SD1.4 DINOv2 cascade | 0.851 | 0.842 in `limited_data` (100%) |
   | GenBuster / BigGAN cascade | 0.899 / 0.995 | 0.898 / 0.998 in `cheap_trigger` and `transfer` |
   | Escalation rate, GenBuster / SD1.4 / BigGAN | 0.259 / 0.266 / 0.007 (`cheap_path`) | 0.275 / 0.275 / 0.006 (`cheap_trigger`) |
   | FFT accuracy, SD1.4 / GenBuster / BigGAN | 0.775 / 0.727 / 0.997 | 0.748 / 0.735 / 0.998 (`cross_family_deployment`) |

   This audit is the checklist for Phase 2.

---

## Phase 1 — Re-analysis that can change headline claims (highest priority)

### 1a. Fix the joint confidence-routing baseline (R1 Comment 8)
**Bug.** In `routing/pipeline/tables/confidence_routing.py`, lines ~205–211 and ~310–333,
inputs that the confidence rule does *not* escalate output `y_fft`, the weakest, stale
probe. That's why the rule scores 0.68–0.75 on SDXL/FLUX. That is *below* cheap-only
(0.82–0.91), even though it escalates about 35% of inputs to a verifier that scores
97–99.6% when run on everything. That result isn't plausible, and R1 will catch it.

**Fix:**
- When the rule doesn't escalate, output the semantic probe's prediction. Report the
  "more-confident probe" variant as a second row.
- For SDXL/FLUX, use the full uniform GPT-5.5 labels (`tab_e8_gpt55_FULL_coverage.csv`
  or the per-sample file behind it) for every escalated sample. Delete the "no cached
  verifier label" fallback and the bounding table in `CONFIDENCE_ROUTING_NOTES.md`.
- Keep the matched-escalation-rate protocol and the validation-split threshold as they
  are.

**Deliverable.** A regenerated `tab_confidence_routing.csv`, plus a new table
`tab:confidence_routing` in Section V (Routing Analysis) of the paper. The table needs:
config, verifier, n, accuracy and escalation rate for both rules, and Δ with a 95% CI.
Add a short method paragraph that defines H_joint, how the threshold is fit, and the
fallback.

**Then:** update the 7/7 claim in C4, the Conclusion (RQ2) and the R1 Comment 8
response to whatever the corrected numbers show. **If disagreement no longer wins
everywhere, say so plainly. Don't tune the baseline until it loses.**

### 1b. Complete the held-out gate analysis (R1 Comments 1–2)
Work from `routing/pipeline/tables/gate_holdout.py` and `tab_gate_holdout.csv`.
1. **Effective N.** Add the actual `calib_n` per configuration for each N setting. N is
   capped at `n_disagree // 2`, so "N=200" is much smaller for small pools; GenBuster
   GPT-5.5 gets 26. Add a caption footnote saying so, the same way R2 asked for on the
   same-pool table, and list which configurations are capped.
2. **Missed-beneficial-escalation rate.** Record how often the oracle says escalate but
   the frozen decision falls back. Report it next to the harmful-escalation rate for
   both rules. A 0.3% harmful rate means little if the rule almost never escalates.
3. **The four positive cells.** For the SDXL/FLUX × {SigLIP2, DINOv2} × GPT-5.5 cells,
   report the fraction of repeats in which the significance-aware gate enables
   escalation at N = 50, 100, 200 and half-pool. This is the key evidence that the gate
   captures the wins.
4. **Rule comparison.** The significance-aware rule agrees *less* often than the naive
   rule at N=200 (24 vs 26) and half-pool (26 vs 29). Discuss that in the text instead
   of only saying "small cost in mean regret".

**Deliverable.** Regenerated table `gate_holdout` with the added rows and columns, and
the revised paragraph in Section VI.I.1.

---

## Phase 2 — Make every table come from one evaluation set (R1 Comment 5, provenance)

1. For each mismatch in `number_audit.md`, find why the scripts disagree. Usual causes
   are a different split, a different probe snapshot, or an "archival subset". Then
   point the script at `routing/pipeline/splits.py`'s canonical split and current
   probes:
   - `cheap_trigger.py`, `transfer.py` and the limited-data and cross-family code in
     `baselines.py`: after the fix, the 100%-fraction row and the in-domain cells must
     equal Table `main_results` exactly.
   - For the cross-family table, FFT-only must equal the in-domain FFT accuracy from
     Table `main_results`, or the caption must explain why it differs.
2. Regenerate `per_generator.py` on the canonical 1,000-sample GenBuster evaluation set,
   using the current, leak-free probes and SigLIP2 as the primary backbone. Drop the
   "400-sample archival subset" caption.
3. Remove the captions that say "fixed held-out subsets … results are interpreted
   within this experiment".
4. Regenerate the two stale figures from current CSVs, `plot_accuracy_comparison.png`
   and `plot_cost_accuracy_all.png`, and fix their `\includegraphics` paths. Lines 701
   and 902 still use the nonexistent `" (1).png"` names.
5. Re-run `baseline_comparison.py` and `gate_analysis.py` after these fixes, since they
   read the same per-cell arrays. Then update Tables `baselines`, `sensitivity` and
   `threeway`, plus every 6/35, 24/35 and 18/9/8 count in the text.

**Acceptance.** Re-running the audit from Phase 0 shows zero unexplained mismatches.

---

## Phase 3 — Add results that exist on disk but aren't in the paper

1. **Robustness (R1 Comment 9).** Build a table from `e7_compression_extended.csv` and
   `e9_postprocessing_robustness.csv`: FFT, SigLIP2 and DINOv2 × {JPEG q, blur σ,
   down/up-scale factor} × dataset. Put it in the Limitations shortcut discussion and
   fix the wrong `\ref{sec:cost_accounting}` at line ~1388.
2. **Audit of the four positive cells (R1 Comment 7).** Extend Table `e8_results`, or add
   a companion table, with one row per SDXL/FLUX GPT-5.5 cell:
   - disagreement n
   - class mix of the disagreement subset (real/fake counts)
   - GPT-5.5 accuracy on that subset
   - best-cheap accuracy on that subset
   - paired-bootstrap 95% CI on the gap
   - coverage (100%)
3. **Token counts (R1 Comment 10).** From the per-call GPT-5.5 files, report the mean
   input and output tokens per call for each dataset in Section VI.J, plus the source of
   the pricing (add a citation). Then either report C_tok where line ~283 says it is
   reported, or soften that sentence.
4. **FFT/DCT spec (R1 Comment 9).** Add the exact feature definition to Implementation
   (Section IV.B), checking each value against the extractor code:
   - 32-bin radial FFT magnitude profile, L2-normalized
   - 8×8-block DCT with the bottom-right 3×3 submatrix pooled into 8 statistics
   - 3 colour-coherence correlations
   - 256×256 resize
   - `LogisticRegression(C=0.1, random_state=42)`

---

## Phase 4 — Small new runs (ask before spending)

1. **GPT-5.5 coverage, in-domain.** Fill the GenBuster (52/259) and SD1.4 (23/532)
   FFT+SigLIP2 disagreement subsets: about 716 calls through `routing/pipeline/batch_gpt55.py`
   using the same corrected parser. Then re-run Phase 1b and Phase 2.5. SD1.4/GPT-5.5
   leaves the "pool too small" exclusion list.
2. **Qwen coverage, SDXL/FLUX.** Fill the remaining 5–7% of the disagreement subsets on
   a local GPU (`run_qwen_sdxl_flux.py`), then drop the fallback caveat from Table
   `e8_results`.
3. **BigGAN distant-shift discrepancy.** The previous version had GPT-5.5 / Qwen-7B /
   Qwen-72B at 0.801 / 0.689 / 0.467. Now they are 0.582 / 0.515 / 0.623, and the
   ranking has reversed. Spend a bounded amount of time (say 1–2 hours) looking for the
   cause: prompt, image preprocessing or resolution sent to the API, which pool was
   sampled, label mapping. Whatever you find, **add a disclosure paragraph** to
   Section V.D.7. R1 has the previous version and will notice. Also rewrite the "scale
   doesn't predict suitability" sentence so it doesn't depend on which model ranked
   first.

---

## Phase 5 — Text fixes in `paper_updates2.tex`

These can be done at any time, but final numbers come from Phases 1–3.

- **Cost.** In the abstract, Section VI.J (~line 1684) and the Figure D caption,
  "at 58–64% of the cost" is wrong. The table gives 36–42% of uniform cost, a 58–64%
  saving. C3 is correct as written.
- **Gate figures in the abstract and C5.** "20–21 … 24–26" mixes the two rules. Report
  the significance-aware rule only (21/32 at N=100, 24/32 at N=200, or the post-Phase-2
  values).
- **Stale contradictions:**
  - ~line 454: "trained on the first 50% of each dataset". Replace with the
    complementary-split description from Section V.A.
  - ~line 601: PASCAL VOC is dropped "because they cause systematic VLM calibration
    failure". Align with Section IV.B's confound argument.
  - ~line 1265: "11 detector-derived features and an explicit latency reward". This
    contradicts the bandit relabeling. Describe the real feature set and say there is
    no reward.
  - ~line 299 says errors are "weakly correlated under distribution shift". The
    agreement-path audit shows correlated errors rise 4–8× under shift. Rewrite.
  - The abstract, intro (~lines 82–84), Related Work (~line 172), ~line 840 and
    Conclusion RQ1/RQ2 overclaim that disagreement is a "parameter-free indicator of
    input difficulty" that "remains reliable under shift" and "requires no
    calibration". Reword to say that disagreement flags candidates where *a* cheap
    detector is unreliable, that the agreement path degrades under shift, and that
    deciding whether to escalate needs labelled calibration (the gate).
- **PASCAL VOC argument (Section IV.B).** Lead with the stronger evidence: cascade gain
  over the stale probe stays positive under all three real-image sources. Note that the
  FFT probe was trained on ImageNet real images, so its PASCAL VOC failures also show
  sensitivity to the real-image source.
- **Uniform-GPT objection.** Add 2–3 sentences to Section VI.J that address it head-on.
  Uniform GPT-5.5 reaches 99.6% for about $13.61 per 4,000 images, so say when the
  cascade is still worth it: throughput or latency at scale, rate limits, self-hosted
  verifiers, and cost at larger volume.
- **Remove meta-commentary** from the manuscript and keep it in the response letter:
  - "An earlier version … A reviewer correctly noted…" (~405–414, ~1439–1442,
    ~1462–1465)
  - the history of earlier latency numbers (~543)
  - "an earlier draft reported 0.917" (Table `transfer` caption)
  - "corrects and strengthens an earlier draft's claim" (~1353)
  - "no prior evaluation in this line of work" (~1668–1672)
- **Latency wording (~line 1721).** It calls the constants both "hardcoded,
  non-measured" and "fixed hardware measurements". Pick the accurate one.
- **Cross-references:**
  - ~line 140 points the Limitations discussion at `sec:three_cases`.
  - Line ~1388 has the robustness `\ref` problem (fixed in Phase 3).
  - Replace hard-coded "Section~VI.I", "V.D.6", "III.D" and so on with `\ref`.
- **Scope.** The Limitations sentence asking readers to read the title as single-frame
  only is awkward. Either narrow the abstract/intro wording to "frame-level" or flag the
  title to the authors; don't change the title without asking.

## Phase 6 — Response letter and final check

1. Update `response_to_reviewers2.md`. Every number must match the final paper. Fix the
   section numbers: the audit is IV.B.1 (not IV.C.1), and the case-mapping figure is in
   VI.B (not VI.A). Mention that R2 Comment 4 asked for a *table*: add a compact table
   next to the figure, generated from `tab_case_mapping.csv`.
2. Compile the `.tex` (`latexmk`, or at least check for undefined refs and missing
   figure files).
3. Do a final pass on the Phase 0 audit and confirm zero mismatches. Then grep the paper
   and the letter for every headline number (7/7, 6/35, 21/32, 24/32, 7.9, 58–64,
   36–42, 99.6, 97.1) and check that each one matches its CSV.
