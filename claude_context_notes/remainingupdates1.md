Context: IEEE Access resubmission. The manuscript is claude_context_notes/paper_updates2.tex and the response letter is claude_context_notes/response_to_reviewers2.md. Work on branch claude/stoic-cannon-tlq9cy: git pull, then git lfs pull.

## Why we're doing this
The paper's only strong positive result (SD1.4 -> SDXL/FLUX, GPT-5.5 escalation, Table e8_results) may be confounded by image size. In routing/results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv, every SDXL/FLUX fake used exactly 1,271 GPT-5.5 input tokens, while the real ImageNet images have a median of 273. A threshold on input_tokens alone separates real from fake with 99.98% accuracy. Fakes are large and square; reals are small ImageNet JPEGs with varying aspect ratios. The paper already shows this kind of shortcut for BigGAN (C6), and its BigGAN 512 px test shows the VLMs' answers depend on input size. We need to test whether the SDXL/FLUX result survives when size is controlled.

## Step 1: measure the confound (CPU, no API)
1. Record native width, height, file format and file size for every image in the SDXL and FLUX benchmarks: the 2,000 fakes each, plus the real images actually used. Do the same for the COCO and original PASCAL VOC real-image variants (routing/results/*_index_coco_real.csv, *_index_original_real.csv).
2. Extend routing/pipeline/robustness/resolution_only_control.py to SDXL and FLUX, using the same protocol it uses for BigGAN/SD1.4/GenBuster: a classifier on width/height/aspect only. Report accuracy for the ImageNet, COCO and PASCAL VOC real-source variants.
3. Write a short summary and show it to me before Step 2.

## Step 2: build a size-matched copy of SDXL/FLUX (CPU)
Apply one identical transform to every image, real and fake: center-crop to a square, resize to 512x512 (bicubic), and save as PNG. Write the copies to a new directory and never overwrite the originals. The goal is that both classes have identical dimensions and format. Note in the write-up that real images still carry their original JPEG history, which is an unavoidable residual difference.

## Step 3: GPT-5.5 on the size-matched images (API, so ask me first)
Run GPT-5.5 on ALL size-matched images (uniform, the same set as CONSOLIDATED_full_sdxl_flux.csv) through routing/pipeline/batch_gpt55.py. Use the same prompt, parser, model, and Batch API settings as the original run.
- BEFORE submitting: send a 20-image smoke test, check that input_tokens is now the same for real and fake, then give me the call count and a cost estimate and WAIT for my approval.
- Afterwards, confirm the input_tokens distributions match across classes.

## Step 4: cheap detectors and cascade on the size-matched images (GPU)
1. Extract FFT, SigLIP2 and DINOv2 features for the size-matched images with the existing extractors.
2. Apply the SAME stale SD1.4-trained probes used for Table e8_results. Do NOT retrain anything.
3. Recompute, for SDXL/FLUX x {SigLIP2, DINOv2}:
   - stale-probe accuracy
   - disagreement rate
   - GPT-5.5 accuracy on the disagreement subset vs the best cheap detector, with a paired-bootstrap 95% CI
   - cascade accuracy and gain over the stale probe, with a paired-bootstrap CI
   - uniform GPT-5.5 accuracy
   - cascade cost share
4. Also report GPT-5.5 real and fake recall, uniform and on the disagreement subset.
5. Put the new code under routing/pipeline/ and write new CSVs with a _sizematched suffix. Wire them into routing/generate_paper_results.py and routing/build_results_bundle.py. Do not change any existing result.

## Step 5: report to me before editing the paper
Show me one table with original vs size-matched results for all four cells. Then:
- If the cascade gain over the stale probe stays significantly positive in all four cells, add the size-matched results to the paper:
  - "size-matched" rows in Table verifier_subset and Table e8_results, or a compact companion table
  - one paragraph in the deployment-shift results
  - one sentence in Limitations
  - a short addition to the letter's R1 Comment 9 response saying we applied the same resolution control to SDXL/FLUX
- If the gain shrinks or disappears in any cell, STOP. Report the numbers and do not edit the abstract, contributions or conclusions. I'll decide the framing.

## Step 6: text fixes (do these regardless)
1. Table gate: the "SDXL/SigLIP2" and "SDXL/DINOv2" rows sit in the Held-out panel after "Rules choose differently". Move them into the bottom "Significance-aware gate escalates (share of draws)" panel with the FLUX rows.
2. "Disagreement reveals/shows which cheap detector is failing" (C4, Section V.C confidence-routing discussion, RQ2 conclusion, letter R1 Comment 8) is overstated: for a single input, disagreement can't say which detector is wrong. Reword everywhere to: with a labeled calibration sample, the disagreement subset shows which cheap detector is failing there (this is what the gate's three-way selection uses).
3. Remove the leftover "difficulty" framing:
   - rename the section "Disagreement as a Difficulty Signal" to "Disagreement as a Routing Signal"
   - "We estimate the input difficulty" -> "We flag inputs for escalation"
   - "based on estimated difficulty" -> "based on disagreement"
   - Discussion intro "separates easy from hard inputs" -> "flags inputs where a cheap detector is unreliable"
4. Probe-degradation sweep: add one sentence explaining that the case table counts Case 3 only when the verifier beats the BEST cheap detector on the disagreement subset with a CI above zero. Several sweep rows beat the semantic probe without meeting that bar.
5. Three-cases discussion, Case 1: note that BigGAN's near-ceiling FFT accuracy is largely the native-resolution shortcut (Limitations).

## Rules
- Never hand-edit numbers in the .tex. Every value comes from a CSV via generate_paper_results.py. After all changes, rebuild paper_results/ and confirm the audit shows 0 mismatches.
- No API spend without my explicit OK.
- Commit after each step with clear messages, and push to claude/stoic-cannon-tlq9cy.