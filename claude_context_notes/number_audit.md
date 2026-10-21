# Number audit — `paper_updates2.tex` vs. regenerated CSVs (Revision 3, Phase 0)

Generated 2026-10-21 on branch `claude/stoic-cannon-tlq9cy` (base commit `051e343`), after
`git lfs pull` and a fresh `python routing/generate_paper_results.py` run.

Legend: ✓ match (to the printed precision) · ✗ mismatch · ⚠ no live source (the printed
value can't be traced to any CSV written by a current, non-archived script) · ~ rounding only.

---

## 1. Regeneration results (Phase 0, step 1)

`generate_paper_results.py` ran cleanly: 15/15 tables `[OK]`, no failures
(`routing/results/pipeline_audit_log.txt`).

**The committed CSVs were stale.** Regeneration changed 6 tracked files:
`tab_baselines`, `tab_cross_family_deployment`, `tab_e8_results`, `tab_limited_data`,
`tab_transfer` and `tab_verifier_comparison`. Several "known mismatches" in the plan were
read off these stale files, so they look different against fresh output (see §2).

**Tables the paper uses that `generate_paper_results.py` does NOT produce:**

| CSV | Writer | Status |
|---|---|---|
| `tab_confidence_routing.csv` | `pipeline/tables/confidence_routing.py` (standalone `__main__`) | not wired in |
| `tab_gate_holdout.csv`, `tab_agreement_path_audit.csv` | `pipeline/tables/gate_holdout.py` (standalone `main()`) | not wired in |
| `tab_case_mapping.csv` | `pipeline/tables/case_mapping.py` | not wired in |
| `tab_cost_accuracy_three_way.csv` (→ Table `cost_accuracy_tradeoff`, e8 GPT-5.5 cascade cells) | **none found** (only read by `experiments/plot_fig_d_cost_accuracy_tradeoff.py`) | orphan |
| `tab_real_image_source_benchmark.csv` (→ Table `real_image_sources`) | **none found** | orphan |
| `tab_e8_qwen_FINAL.csv` (→ Table `e8_results` Qwen cells) | **none found** | orphan |

The plan's Phase 2/8 goal of "one entry point produces every number" needs all six wired
in. The three orphans need a generator recovered from `_archive/` or rewritten.

---

## 2. The plan's known mismatches, re-checked

| Quantity | Plan said | Fresh CSVs | `.tex` | Verdict |
|---|---|---|---|---|
| SD1.4 cascade (SigLIP2) | main 0.874 vs 0.867 elsewhere | main 0.8735; cheap_trigger 0.8735; transfer 0.8735 | cheap_trigger 0.867, transfer 0.867, limited_data(100%) 0.867 | ✗ tex stale; CSVs now agree with each other |
| SD1.4 DINOv2 sem-only | 0.877 vs 0.890 | main 0.8765; `tab_limited_data.csv` has **no sem-only column** | limited_data 0.890 | ⚠ limited_data sem-only/cascade/gain columns have no live source |
| SD1.4 DINOv2 cascade | 0.851 vs 0.842 | main 0.851; no limited_data cascade column | limited_data 0.842 | ⚠ same |
| GenBuster / BigGAN cascade | 0.899/0.995 vs 0.898/0.998 | cheap_trigger 0.899/0.995; transfer diagonal 0.899/0.992 (classifier) | cheap_trigger 0.898/0.998; transfer 0.899/0.998 | ✗ tex stale |
| Escalation rate GB/SD/BG | 0.259/0.266/0.007 vs 0.275/0.275/0.006 | cheap_trigger 0.259/0.266/0.007 | cheap_trigger 0.275/0.275/0.006 | ✗ tex stale |
| FFT accuracy SD/GB/BG | 0.775/0.727/0.997 vs 0.748/0.735/0.998 | `tab_cross_family_deployment.csv` has **no FFT-only column** (its `best_cheap_acc` is FFT on the disagreement subset, a different quantity) | cross_family FFT 0.748/0.735/0.998 | ⚠ cross_family FFT/Stale/Casc/Δ columns have no live source |

**Takeaway:** across the fresh CSVs, the in-domain numbers are now consistent with
`main_results`. The remaining problems are (a) tex tables printing values from an older
pipeline state, and (b) three tex tables whose printed columns have no generator at all.
Phase 2 should fix the **generators** for (b), not just the CSV-to-CSV agreement.

---

## 3. Table-by-table

### `main_results` (tex l.720–730 ← `tab_main_results.csv`)
| Cell | tex | CSV | |
|---|---|---|---|
| CLIP, GenBuster | 0.904 | 0.943 | ✗ **tex has the leaky probe's value** |
| Casc.(C), GenBuster | 0.876 | 0.891 | ✗ **tex has the leaky probe's value** |
| Casc.(C/S/D) latency, GenBuster | 55.1 / 57.9 / 58.4 | 62.0 / 62.2 / 64.3 | ✗ |
| Casc.(C/S/D) latency, SD1.4 | 36.9 / 31.9 / 32.5 | 34.2 / 35.0 / 37.0 | ✗ |
| Casc.(C/S/D) latency, BigGAN | 13.3 / 13.2 / 13.5 | 13.4 / 13.4 / 14.0 | ✗ |
| Casc.(D), BigGAN | 0.991 | 0.9905 | ~ rounding (half-up vs. float) |
| all other cells | | | ✓ |

*CLIP root cause (verified):* recomputing on the canonical GenBuster eval set,
`clip_preds.csv.leaky_backup` gives exactly 0.904 / Casc.(C) 0.876 (esc 0.271), and the
clean `clip_preds.csv` written by `pipeline/retrain_clip.py` gives 0.943 / 0.891
(esc 0.258). The table was never updated after the leakage fix. The response letter
claims the fix, so this has to be corrected.

*Latency root cause:* CSV latency = cheap constants + esc_rate × Qwen always-escalate
constant (e.g. GB Casc.(S): 2.5 + 10.4 + 0.259 × 190.2 = 62.16). The tex values imply
older escalation rates or a different Qwen constant. They're stale.

### `verifier_comparison` (l.744–751) — ✓ all cells.
### `baselines` (l.780–782) — ✓ 28/35, 22/35, 6/35. The six named configurations in the text (l.787–790) ✓. Inline l.793–796 (58.6%, N=532, 50.0% [45.9, 54.3], 52.8%, 78.4%, [−0.250, −0.145]) ✓.

### `e8_results` (l.813–814)
- GPT-5.5 Acc/Gain ✓ against `tab_cost_accuracy_three_way.csv` / `tab_e8_gpt55_FULL_coverage.csv`, but ⚠ both are orphans (§1).
- Qwen Acc/Gain ✓ against `tab_e8_qwen_FINAL.csv` (⚠ orphan).
- `tab_e8_results.csv` (the live output) holds only verifier-on-disagreement accuracy, so none of the printed cells come from it.
- **New:** the live `tab_e8_results.csv` / `tab_baselines.csv` / gate analysis evaluate the GPT-5.5 cells on n=1354/1450/1594/1552, which is the **Qwen-covered intersection**. The full GPT-5.5 disagreement subsets are 1452/1539/1675/1659. So the "100% GPT-5.5 coverage" claim holds for the cascade cells, but not for the baselines/gate cells that support the four positive results. Fix in Phase 2 (and use the full subsets in the Phase 3.2 positive-cell audit).

### `escalation` (l.857–860) — ✓. `cheap_path` (l.931–933) — ✓.

### `per_generator` (l.880–885 ← `tab_per_generator.csv`)
| Generator | tex Casc. | CSV Casc. | tex Real CLIP | CSV |
|---|---|---|---|---|
| CogVideoX | 0.822 | 0.589 | | |
| EasyAnimate | 0.918 | 0.781 | | |
| HunyuanVideo | 0.875 | 0.625 | | |
| LTX-Video | 0.879 | 0.727 | | |
| Real | 0.922 | 0.980 | 0.980 | 0.980 |

✗ Every cascade cell differs. CNNSpot/CLIP columns ✓. The CSV is labelled
`descriptive_400_subset_old_small`. Already in Phase 2.2.

### `cheap_trigger` (l.973–975 ← `tab_cheap_trigger.csv`) — ✗ and **conclusion-changing**
| | tex esc / disag / random [CI] / beats | CSV |
|---|---|---|
| GenBuster | 0.275 / 0.898 / 0.904 [0.880, 0.925] / No | 0.259 / 0.899 / 0.723 [0.707, 0.740] / **True** |
| SD1.4 | 0.275 / 0.867 / 0.839 [0.817, 0.860] / Yes | 0.266 / 0.8735 / 0.722 [0.709, 0.735] / Yes |
| BigGAN | 0.006 / 0.998 / 0.994 [0.990, 0.996] / Yes | 0.007 / 0.995 / 0.995 [0.993, 0.997] / **False** |

**New bug (same class as Phase 1a):** `pipeline/tables/cheap_trigger.py:59`
initializes the random-trigger prediction to `y_fft`. Non-escalated inputs therefore get
the FFT prediction, including the disagreement cases the random mask skips, where FFT is
3.9% accurate on GenBuster. This makes the random baseline artificially weak
(0.723 ≈ FFT-only 0.727). A fair random baseline falls back to the semantic probe. The
text (l.949–960) and response letter (l.369: "0.898 vs. 0.904") describe the semantic
fallback result. **Add to Phase 1:** fix the fallback, then reconcile the table and text.

### `limited_data` (l.1000–1012) — ⚠ entire Sem.-only / Cascade / Gain table has no live source; `tab_limited_data.csv` contains verifier-vs-baselines rows only. Needs a generator (Phase 2.1).

### `transfer` (l.1046–1050 ← `tab_transfer.csv`)
| Train→Test | tex GB / BG / SD | CSV GB / BG / SD |
|---|---|---|
| GenBuster | 0.899 / 0.998 / 0.867 | 0.899 / 0.995 / 0.8735 |
| BigGAN | 0.843 / 0.998 / 0.758 | 0.854 / 0.992 / 0.793 |
| SD14 | 0.897 / 0.998 / 0.867 | 0.899 / 0.995 / 0.8735 |
| Rule | 0.899 / 0.998 / 0.867 | (main) 0.899 / 0.995 / 0.8735 |

✗ 8 of 12 cells. Note: the fresh BigGAN in-domain classifier (0.992) is *below* the
rule (0.995). The text at l.1056–1061 (0.900 vs. 0.903, CI [−0.0100, 0.0000]) ⚠ has no
CSV source.

### `cross_family_deployment` (l.1125–1132) — ⚠ no live source for FFT / Stale / Casc. / Δ columns (§2).

### `biggan_shift` (l.1182–1184) — ✓ (0.582 / 0.515 / 0.623; 0.907). The discrepancy with the *previous* paper version is Phase 4.3.

### `real_image_sources` (l.509–524) — ✓ against `tab_real_image_source_benchmark.csv` and the e8 cells, but ⚠ orphan.

### `sensitivity` (l.1555–1556) — ✓ (13/19/21/24/28 and 12/19/21/24/28). `threeway` (l.1592–1594) — ✓ 18/9/8.

### `gate_holdout` (l.1572–1575) — ✓ all eight agreement counts and harmful rates; 32 eligible / 3 excluded ✓. CSV is produced standalone only (§1).

### `cost_accuracy_tradeoff` (l.1703–1706) — ✓ against `tab_cost_accuracy_three_way.csv` (⚠ orphan).

### `confidence_routing` — no table in the paper (Phase 1a adds one). Current CSV: 7 configs, all Δ < 0, but in-domain GenBuster conf-rule acc 0.760 and SDXL/FLUX 0.68–0.75 show the `y_fft` fallback bug. Also: in-domain GenBuster threshold fit on **n_subval = 99**.

---

## 4. Inline numbers

| Location | Quoted | Source | |
|---|---|---|---|
| Abstract, C3 | up to 7.9 pp | e8 GPT-5.5 max gain 0.0785 (SDXL/DINOv2) | ✓ |
| Abstract, VI.J l.1684, Fig. D caption l.1713 | "at 58–64% of the cost" | cost ratio 36.3–41.9%; saving 58.1–63.7% | ✗ wording (Phase 5) |
| C3 | invocations reduced 58–64% | esc rate 0.363–0.419 | ✓ |
| Abstract | uniform 97–100% | 0.9705 / 0.996 | ✓ |
| C3, l.1686 | 3–7.8 pp remain | 0.0303–0.078 | ✓ |
| Abstract, C5 | 20–21/32 @N=100, 24–26 @N=200 | naive 20 / sig 21; naive 26 / sig 24 | ✓ numbers, ✗ mixes rules (Phase 5) |
| C5, abstract | order-of-magnitude harmful reduction | 5.5% → 0.2% etc. | ✓ |
| C4, l.1736 | 7/7 confidence configs | `tab_confidence_routing.csv` | ✓ vs. CSV, but CSV buggy (Phase 1a) |
| l.834–839 | 3.9%, 21.6%, 96.1%, 78.4%, 92.6%, "above 96%" | cheap_path, verifier_comparison, main | ✓ |
| l.951–955 | 0.867 vs 0.839; 0.998 vs 0.994; 0.898 vs 0.904 | cheap_trigger | ✗ (see §3) |
| l.1056–1061 | 0.899; 0.900 vs 0.903; CI [−0.0100, 0.0000] | none found | ⚠ |
| l.1350–1352 | class-wise recalls (95.1/91.4 → 98.3/97.5, …) | `pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv` | ✓ (spot-checked FLUX + SDXL cascade) |
| l.1360–1361 | 2.4% GenBuster; 8.2–16.1% SDXL/FLUX agreement path | `tab_agreement_path_audit.csv` (24/1000; 0.082–0.161) | ✓ |
| l.1379–1400 | 99.7, 99.55, 0.15 pt, 50.1, 99.65 @q50, 29.2 @σ2, 33.9 @4×, 92.8, 77.5, 63.9, 50.0 | resolution_only_control, resolution_matched_ablation, e7_compression_extended, e9_postprocessing_robustness | ✓ (63.95 printed as 63.9) |
| l.1491–1495 | agreement-path 96.8/3.2, 97.8/2.2, 99.8/0.2, 88.4/11.6, 88.7/11.3, 91.8/8.2, 83.9/16.1 | `tab_agreement_path_audit.csv` | ✓ |
| l.1519–1531 | 18/21/24 sig; 17/20/26 naive; 21/24/28 oracle; 6.7% vs 0.3% | gate_holdout, sensitivity | ✓ |
| l.1614–1630 | 13/35 vs 12/35; A₁=0.216, A₂=0.784, A₃=0.587; 35/1000; 3.52%; 89.3% vs 91.6% | sensitivity/threeway + `gate_rules/explain_n10_divergence.py` | ✓ counts; derivation not re-run |
| l.1675–1684 | $2.50/$15.00; 8,000 calls; 99.6%, 97.1%; $13.61, $18.47; 92.5–93.0%, 89.3–94.0%; 36–42% esc | cost_accuracy_three_way, cost_accounting | ✓ |
| l.1721–1722 | latency constants | `main_results.py` constants | ✓ |

Not yet audited line by line: Related Work, Method (Sections III–IV prose), and the
response letter. These go into the Phase 8 `inline_numbers` sheet.

---

## 5. Changes to the plan this audit implies

1. **Phase 1 gains an item, 1c: fix the `cheap_trigger.py:59` random-baseline fallback** (`y_fft` → semantic). It flips the GenBuster/BigGAN "beats random" verdicts between the CSV and the paper, so it is headline-relevant.
2. **Phase 2 gains a CLIP item:** update `main_results` CLIP/Casc.(C) GenBuster to the clean-probe values (0.943 / 0.891) and the cascade latencies. Check every downstream use of CLIP GenBuster (Fig. accuracy comparison, per_generator CLIP column).
3. **Phase 2.1 is about missing generators, not just mismatched ones.** `limited_data` (sem-only/cascade/gain), `cross_family_deployment` (FFT/stale/casc/Δ) and the transfer robustness numbers need code that writes the printed columns.
4. **Phase 2/8: wire in** `confidence_routing`, `gate_holdout` and `case_mapping`, and recover or rewrite generators for the three orphan CSVs.
5. **Phase 1b/3.2: GPT-5.5 positive cells** should be evaluated on the full GPT-5.5 disagreement subsets (1452/1539/1675/1659), not the Qwen-covered intersection (1354/1450/1594/1552).
6. **Phase 1a line anchors:** the `y_fft` fallbacks are at `confidence_routing.py:211` (in-domain) and `:326–334` (SDXL/FLUX). Semantic predictions are `y_siglip2` and `y_sem` respectively. The SDXL/FLUX "no verifier label" gap is `:311–334`; the bounding table is `CONFIDENCE_ROUTING_NOTES.md:71–80`.
7. **Phase 1b scope shrinks:** `calib_n` and `capped_below_target` already exist in `tab_gate_holdout.csv`, and the per-N escalate fraction for the four positive cells is already the `sig_escalation_rate` column. Still new: the missed-beneficial-escalation rate, and a per-cell naive-vs-sig disagreement record (`gate_holdout.py:122–155`).
