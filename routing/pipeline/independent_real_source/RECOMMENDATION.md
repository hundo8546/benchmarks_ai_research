# Independent real-image source: findings and recommendation

Context: Reviewer 1, point 4 (`claude_context_notes/peerreviewdecision2.txt`) asks for (a) results on
the ORIGINAL AIGIBench PASCAL VOC real-image benchmark alongside the modified (ImageNet/SD1.4) one,
and (b) evaluation on an independently-chosen real-image source distinct from both PASCAL VOC and the
ImageNet/SD1.4 pool already reused across SD1.4/SDXL/FLUX ("varying compression, resolution, and
acquisition history"). This note covers (b).

## Candidates considered

### 1. COCO val2017 — RECOMMENDED, obtained and evaluated

- **Not present on disk beforehand** (`find / -iname "*coco*" ...` found nothing but stdlib/package
  files unrelated to the dataset).
- **Internet access confirmed working** in this environment (`pip download requests` succeeded;
  TLS/DNS to google.com fine). One wrinkle: `https://images.cocodataset.org/...` (virtual-hosted-style
  S3 URL) fails TLS verification here (curl error 60 — the certificate returned is for
  `s3.amazonaws.com`, which doesn't match the `images.cocodataset.org` SNI/Host). The **path-style**
  URL works fine: `https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip` (confirmed
  `HTTP 200`, `Content-Length: 815585330`).
- **Obtained**: downloaded `val2017.zip` (815MB) in 26.6s (~30MB/s) into its own fresh directory
  (`data/_downloads_coco/`, per untrusted-download hygiene), then extracted (standard `val2017/*.jpg`
  structure, no path-traversal concerns) to `data/independent_real/val2017/` — 5000 images, 781MB total.
  Confirmed varied resolution (e.g. 640x428, 500x375, 427x640) and varied file size (70KB–224KB),
  i.e. genuinely different compression/resolution/acquisition history than both PASCAL VOC (old,
  heavily downsampled/compressed) and the ImageNet val/nature pool (SD1.4's own eval source).
- **License**: Flickr-sourced images under a mix of CC BY 2.0/3.0/4.0, CC0, public-domain and
  US-Government licenses (per-image attribution available in COCO's annotation files); the COCO
  Consortium's own terms permit research use of the images. No registration/auth required for the
  image zips themselves.
- **Time/size**: well under the 1-hour / few-GB budget (download+extract took under 2 minutes total).
- Sampled 2000 images per generator (seed=42, same sampling logic as the PASCAL-VOC swap script),
  paired with the EXISTING SDXL/FLUX fake images (unchanged), extracted FFT/SigLIP2/DINOv2 features,
  and ran the same stale-probe disagreement analysis as Part 1. See exact counts below.

### 2. RAISE (or Dresden Image Database) — NOT pursued, documented for completeness

- **RAISE**: site is reachable (`http://loki.disi.unitn.it/RAISE/`, `HTTP 200`). It offers 8156
  **raw, camera-native, uncompressed** images (TIFF/RAW, never processed) collected by 4 photographers
  over 2011–2014 — excellent forensic provenance (true camera-raw acquisition history), but each file
  is tens of MB (a single flat-field add-on pack alone is ~5GB). A 2000–4000 image sample at that
  per-file size would plausibly be 60–160GB+ and take well beyond the ~20–30 min compute/time budget
  for this step, even before considering that RAW/TIFF files would need format conversion before
  running the existing FFT/SigLIP2/DINOv2 feature extractors. Not attempted.
- **Dresden Image Database**: not confirmed reachable in this environment in the time available
  (a direct connection attempt to `forensics.inf.tu-dresden.de` timed out); independent of reachability,
  it is historically distributed under a data-use request/agreement rather than a direct anonymous
  download, which is not something that can be completed as a local/free, no-registration step here.
  Not attempted.
- Neither was ruled out for all time — if the paper specifically wants camera-RAW provenance (a
  stronger claim than COCO's Flickr JPEGs), RAISE is the better-documented option, but it requires a
  separate budgeted session for bulk RAW download + format conversion, which is out of scope for this
  local/free, time-boxed step.

## Exact disagreement counts obtained (n_total = 4000 = 2000 real + 2000 existing fake per row)

All numbers computed 100% locally (stale SD14-trained FFT/SigLIP2/DINOv2 probes applied zero-shot);
zero GPT-5.5 / OpenAI API calls were made to produce any of these counts.

| source   | dataset | backbone | n_disagree | disagree_rate | fft_stale_acc | sem_stale_acc | frac real→FFT="fake" | frac real→sem="fake" |
|----------|---------|----------|-----------:|---------------:|---------------:|---------------:|----------------------:|----------------------:|
| original (PASCAL VOC) | sdxl | SigLIP2 | **2198** | 0.5495 | 0.3158 | 0.7143 | 0.833 | 0.348 |
| original (PASCAL VOC) | sdxl | DINOv2  | **2227** | 0.5568 | 0.3158 | 0.7130 | 0.833 | 0.363 |
| original (PASCAL VOC) | flux | SigLIP2 | **2401** | 0.6003 | 0.2655 | 0.7543 | 0.846 | 0.359 |
| original (PASCAL VOC) | flux | DINOv2  | **2351** | 0.5878 | 0.2655 | 0.6773 | 0.846 | 0.370 |
| coco (independent)    | sdxl | SigLIP2 | **1434** | 0.3585 | 0.6378 | 0.8598 | 0.189 | 0.057 |
| coco (independent)    | sdxl | DINOv2  | **1532** | 0.3830 | 0.6378 | 0.8383 | 0.189 | 0.112 |
| coco (independent)    | flux | SigLIP2 | **1624** | 0.4060 | 0.6008 | 0.8998 | 0.175 | 0.068 |
| coco (independent)    | flux | DINOv2  | **1641** | 0.4103 | 0.6008 | 0.8055 | 0.175 | 0.113 |

Source CSVs: `routing/results/disagreement_original_real.csv`, `routing/results/disagreement_coco_real.csv`.

### Sanity check vs. the paper's claimed 82–97% GPT-5.5 misclassification on original PASCAL VOC reals

- The local FFT artifact detector calls **83–85%** of the original PASCAL VOC real images "fake" —
  strikingly consistent with the paper's cited 82–97% GPT-5.5 misclassification range, even though FFT
  is a completely different (frequency-domain, non-VLM) detector. This independently corroborates that
  PASCAL VOC's old, low-resolution, heavily-JPEG-compressed photos carry a genuine artifact-level
  confound that both artifact detectors and GPT-5.5 pick up on — not just a GPT-5.5-specific failure.
- The stale SigLIP2/DINOv2 semantic probes call only **35–37%** of the same images "fake" — much lower,
  confirming (as expected and noted as acceptable in the task brief) that semantic-embedding detectors
  do not replicate GPT-5.5's failure mode the way the artifact detector does.
- On COCO, by contrast, the FFT detector calls only **17–19%** "fake" and semantic probes call only
  **6–11%** "fake" — both far lower than on PASCAL VOC, consistent with COCO being a cleaner, more
  modern-acquisition real-image pool with fewer compression artifacts that could be mistaken for
  generative artifacts.

## Recommendation

**Use COCO val2017** as the independent real-image source for the reviewer-requested evaluation. It is
already downloaded, license-clear, exhibits genuinely different compression/resolution/acquisition
history from both PASCAL VOC and the ImageNet/SD1.4 pool, and required no registration or format
conversion. RAISE remains a documented fallback if the paper later wants true camera-RAW provenance,
but it needs a separately budgeted download/conversion pass (tens of GB, RAW→standard-format
conversion) that doesn't fit this local/free, time-boxed step.

## Exact API calls needed for a full reproduction (counted, not guessed)

Per (dataset × semantic backbone) combination, the number of NEW GPT-5.5 calls needed equals
`n_disagree` from the table above — these are the disagreement-subset sizes a verifier would be run on,
exactly mirroring how `e8_modern_diffusion.py --run_gpt55` already operates on the current
ImageNet-real benchmark. If both SigLIP2 and DINOv2 pipelines are wanted for a given (source, dataset),
the counts are NOT simply additive in general (the two disagreement subsets overlap but are not
identical) — summing them is a conservative upper bound; the union of paths needing a label is <= the
sum. Totals:

- **Original PASCAL VOC benchmark, SDXL+FLUX, SigLIP2+DINOv2 (naive sum, conservative upper bound)**:
  2198 + 2227 + 2401 + 2351 = 9177 calls.
- **COCO independent-source benchmark, SDXL+FLUX, SigLIP2+DINOv2 (naive sum, conservative upper bound)**:
  1434 + 1532 + 1624 + 1641 = 6231 calls.

The naive per-backbone sum double-counts images that appear in BOTH the SigLIP2-pipeline and
DINOv2-pipeline disagreement subsets (if a single GPT-5.5 call's REAL/FAKE label is reused across both
pipelines, which is the natural thing to do — the call doesn't depend on which semantic backbone
disagreed with FFT, only on the image itself). We computed the exact de-duplicated UNION of
disagreeing images per (source, dataset), which is the real number of GPT-5.5 calls actually needed:

| source   | dataset | union n_disagree (SigLIP2 ∪ DINOv2) |
|----------|---------|---------------------------------------:|
| original (PASCAL VOC) | sdxl | **2839** |
| original (PASCAL VOC) | flux | **3001** |
| coco (independent)    | sdxl | **1892** |
| coco (independent)    | flux | **2034** |

- **Original PASCAL VOC benchmark, SDXL+FLUX, exact union**: 2839 + 3001 = **5840 calls**.
- **COCO independent-source benchmark, SDXL+FLUX, exact union**: 1892 + 2034 = **3926 calls**.
- **Grand total for both reviewer-requested evaluations, exact union**: **9766 calls**.

(The naive-sum totals above, 9177 and 6231, are upper bounds if the two semantic pipelines are instead
treated as fully separate experiments each re-querying GPT-5.5 on their own disagreement subset without
label reuse across pipelines — e.g. if they're reported and audited as independent runs. Which number
applies depends on whether the next budget-gated step reuses one GPT-5.5 label per image across both
backbones (use the union counts) or runs each backbone's disagreement subset as a fully separate API
pass (use the naive-sum counts).)
