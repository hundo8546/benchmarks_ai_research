# Recommendations for Next Session

Written at the end of the 2026-07-17 session that addressed the remaining Reviewer 2/3
items from `remaining_work.md` (bootstrap CIs, cheap-trigger ablation, open-weight
verifier, BigGAN distant-generator test) and did a full compile verification against
the real Overleaf project assets. All work from that session is in PR
[#2](https://github.com/hundo8546/benchmarks_ai_research/pull/2) on branch
`revision-session-updates` — merge that first if you haven't.

---

## Part 1: Tables to remove or merge (currently 14 total)

Reviewer 1's original complaint was "too many tables" (18 in the reviewed version,
already cut to 14). If you want to cut further, here's every table with a concrete
recommendation, ranked by how strong the case is.

### Cut these first — weakest justification for existing at all

- **Table 9 (`tab:matched_entropy`)** — Disagreement vs. entropy routing, but the
  caption states it explicitly: *"CNNSpot+CLIP configuration"* — the secondary/legacy
  backbone, not the primary FFT+SigLIP2 one the rest of the paper uses. The same
  disagreement-vs-entropy comparison already exists on the **primary** backbone via
  the "Entropy thresh" rows already in Table 2. This table only adds "the effect also
  holds on a different backbone," which is a robustness footnote, not new evidence.
  **Why this matters beyond table count**: keeping a table on an inconsistent backbone
  invites exactly the kind of "which backbone is this?" scrutiny Reviewer 1 already hit
  the paper on once (Table 15 vs. 16 unfair comparison). Removing it removes that risk
  entirely rather than just hiding it.

- **Table 13 (`tab:clip_transfer`)** — Cross-dataset transfer matrix for the CLIP probe
  specifically, not SigLIP2 (the primary backbone). The phenomenon it shows
  (semantic probes don't transfer across generator families) is already carried by
  Table 14's "Stale SigLIP2" column plus its caption's in-domain-ceiling numbers.
  Same backbone-inconsistency risk as Table 9.

### Cut or merge — redundant with data already stated elsewhere

- **Table 5 (`tab:e8_efficiency`)** — Two columns (disagreement rate, token savings),
  and both numbers are already stated in the abstract and Practical Implications
  prose ("58–65% of verifier invocations"). Either cut entirely or fold as two extra
  columns onto Table 4.

- **Table 6 (`tab:crossdataset`)** — Re-presents "gain over FFT-only" using numbers
  that are already all present in Table 2 (FFT-only, SigLIP2-only, and
  Disagree→Qwen rows exist there per dataset; gain is just a subtraction a reader—or
  you—could add as a column to Table 2 instead of a standalone table).

### Move to supplementary rather than cut — the data is genuinely unique, just narrow

- **Table 8 (`tab:per_generator`)** — Real, unique per-generator accuracy breakdown
  on GenBuster (Gen3, Jimeng, Kling, etc.), not restated anywhere else. Worth keeping
  in the paper somewhere, but it's a narrow, single-benchmark drill-down — a natural
  supplementary-material candidate if page count is tight.

- **Table 10 (`tab:cheap_path`)** — Agreement-subset coverage/accuracy per dataset.
  Distinct from Table 7 (which is GenBuster-only case-type breakdown), but both are
  "case-behavior on the cheap path" tables. Consider merging them into one table
  (Table 7's 4-row case breakdown + Table 10's 3-dataset coverage numbers) rather than
  keeping as two.

### Keep as-is, lowest priority to touch

- **Table 1 (`tab:litsurvey`)** — Standard related-work comparison table, reviewers
  expect this in the intro. Least objectionable of the seven "optional" tables.

### Do not touch — these are the paper's actual evidence

Table 2 (`main_results`), Table 3 (`verifier_comparison`), Table 4 (`e8_results`),
Table 7 (`escalation`), Table 11 (`limited_data`), Table 12 (`transfer`),
Table 14 (`cross_family_deployment`). Each is the *only* place a specific claim's
evidence lives — removing any of them removes the evidence for a claim the paper
makes, not just a redundant restatement.

**If cutting for real**: Table 9 + 13 first (14→12, removes the backbone-inconsistency
risk). Add Table 5 + 6 if you need to hit 10 (14→10, matching the original
`remaining_work.md` target).

---

## Part 2: Paper edits — what I'd change and why

### High priority — the Abstract and Contributions are now stale

This is the most important item in this whole document. The Abstract and
Contributions (C1–C4, `updatedpaper.tex` lines ~59 and ~112–118) still only describe
the SDXL/FLUX positive result. They do **not** mention:

- The **Verifier Validation Gate** (Section VI.I) — a full practical
  fail-safe mechanism, retrospectively validated at 20/21 (95.2%) configurations.
  This is a real, standalone contribution and currently has no corresponding "C5."
- The **BigGAN distant-generator finding** (Section V.D.7) — that escalation fails
  under a genuinely distant (GAN) shift even with GPT-5.5, the strongest verifier
  tested. This is arguably the single most scientifically interesting result added
  this session, and it's invisible from the abstract.
- The **open-weight verifier test** (Qwen2.5-VL-72B) and its mixed (SDXL yes,
  FLUX no) result.

Why this matters: a reviewer reading only the abstract would think the paper's
positive claim is broader/more universal than what the body actually demonstrates.
The body is now *more* honest and *more* conditional than the abstract — that's
backwards from how it should read, and it's exactly the kind of gap an Associate
Editor doing a final pass would flag. Recommend:
1. Add a **C5** contribution stating the Validation Gate result.
2. Revise the abstract's last two sentences to say the cascade's benefit is
   conditional on verifier strength and shift distance (tested and characterized,
   not assumed), and name the BigGAN negative result as evidence *for* that
   characterization rather than a limitation to hide.

### Medium priority — response letter format doesn't match IEEE's requested template

The rejection email explicitly asks for a response letter with three distinct parts
per point: **(a)** reviewer's concern, **(b)** your response, **(c)** your action
taken. `response_to_reviewers.txt` currently uses a two-part REVIEWER/RESPONSE
structure that blends (b) and (c) together. Not a content problem — everything
needed is in there — but worth reformatting into the explicit three-part structure
before resubmission so it matches what was asked for, since IEEE Access states
non-compliant resubmissions can be rejected without further review.

### Medium priority — a residual scope gap worth one sentence

Reviewer 2 suggested testing "a GAN→diffusion shift, **or** an autoregressive image
generator." This session tested the GAN direction (BigGAN) only — a legitimate choice
since the reviewer offered either as sufficient, but currently the paper doesn't say
this explicitly anywhere. Recommend one sentence in Limitations: something like "we
tested distributional distance via generator architecture (GAN); an autoregressive
generator family remains untested and could show a different failure mode." This
preempts a "you only tested one of the two things I suggested" follow-up rather than
leaving it for a reviewer to notice unprompted.

### Low priority — page count and prose density

The paper grew from 14 to 21 pages across the full revision (not just this session).
Overfull/underfull box rate per page is actually *slightly better* than the original
(confirmed via real compile), so this isn't sloppiness, just genuine volume. If page
count matters for this venue, the newer sections I wrote this session (Cheap-Trigger
Ablation, BigGAN distant shift, the AIGIBench class-breakdown paragraph in
Limitations) are reasonable places to tighten — I erred toward being thorough and
explicit about mechanism in each, which cost some length.

### Administrative, not editorial — must happen before resubmission regardless

- **`access.tex` on Overleaf is still the old, pre-revision file.** This session only
  pushed `tracked_revisionv2.tex` (the diff) to the Overleaf project — `access.tex`
  itself was intentionally left untouched, since only the tracked-changes file was
  requested. Before resubmission, `access.tex` needs to be replaced with the current
  `updatedpaper.tex` content, and a real "Highlighted PDF" needs to be compiled from
  `tracked_revisionv2.tex` in Overleaf (real fonts/branding aren't reproducible in
  this sandbox — confirmed the licensed Formata/GiovanniStd fonts only render
  correctly in Overleaf's own environment).
- **PR [#2](https://github.com/hundo8546/benchmarks_ai_research/pull/2)** on the
  `benchmarks_ai_research` repo needs review/merge — it has all of this session's
  scripts, result CSVs, and paper/response-letter edits.
- Two real (pre-existing, not introduced this session) table-layout bugs were found
  and fixed during the compile verification: Table 4 and Table 6 were both using a
  font size too large for their column width, causing 0.7"+ overflow into the page
  margin. Already fixed and pushed — no action needed, just noting it was a genuine
  bug independent of anything in this document.
