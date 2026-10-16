# routing/pipeline — Phase 0 remediation modules

Factual reference for the FFT/DCT artifact-detector feature definition and
classifier hyperparameters, transcribed directly from
`routing/extractors/extract_fft.py` (re-read in full for this document, not
paraphrased from memory) for the peer-review response. Precision over
prose.

## FFT/DCT feature vector — exact definition

Total dimension: **43** (confirmed empirically against `fft_features.csv`'s
43 `f*` columns and a live re-run of `extract_features()` on a dummy image).

Preprocessing, in order:
1. `img_pil.resize((256, 256))` — PIL `Image.resize`, `resample` argument
   **not specified** (defaults to Pillow's `BICUBIC` for non-palette/non-1-bit
   images). Target size is fixed at 256x256 regardless of source aspect
   ratio (non-aspect-preserving resize).
2. Convert to RGB uint8 numpy array (`img_rgb`, shape `(256, 256, 3)`).
3. Grayscale via `cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)` → `gray`
   (uint8, shape `(256, 256)`).
4. `gray_f32 = gray.astype(np.float32) / 255.0` — a *separate*,
   `[0,1]`-normalized float copy, used **only** by the FFT stream (step A
   below). The DCT stream (step B) uses the raw uint8 `gray`, not
   `gray_f32`.

### Stream A — radial FFT magnitude profile (32 dims)

```python
f = np.fft.fft2(gray_f32)
fshift = np.fft.fftshift(f)
mag = np.log1p(np.abs(fshift))
```
- `N_RADIAL_BINS = 32`.
- For each pixel, radius `r = sqrt((y-cy)^2 + (x-cx)^2)` from the spectrum
  center `(cy, cx) = (h//2, w//2)` (h=w=256), normalized by
  `r_max = sqrt(cy^2 + cx^2)` to `r_norm in [0, 1]`.
- 32 equal-width bins over `r_norm in [0, 1]` (`np.linspace(0, 1, 33)`).
  `profile[i] = mean(mag[bin_i])` (bins with zero pixels stay 0).
- Final step: **L2-normalize** the 32-dim profile vector
  (`profile / (||profile||_2 + 1e-8)`).

### Stream B — DCT high-frequency block statistics (8 dims)

```python
DCT_PATCH = 8
img = gray[:ph, :pw].astype(np.float32) - 128.0   # ph,pw = gray cropped to multiples of 8
```
- Image cropped (not padded) to the largest multiple of `DCT_PATCH=8` in
  each dimension, then **mean-centered by subtracting 128** (not per-block,
  one global offset).
- Non-overlapping 8x8 blocks; `cv2.dct(block)` per block (standard 2-D DCT-II).
- **High-frequency submatrix = the bottom-right 3x3 of each 8x8 DCT block**,
  i.e. `dct[5:, 5:]` (rows 5,6,7 x cols 5,6,7, 0-indexed) — the 9
  highest-frequency coefficients, flattened and pooled **across all blocks
  in the whole image** into one long 1-D array `hf`.
- 8 summary statistics computed over the pooled `hf` array:
  `mean, std, skew, kurtosis, p5, p25, p75, p95` (skew/kurtosis via
  `scipy.stats.skew` / `scipy.stats.kurtosis`; percentiles via
  `np.percentile`).

### Stream C — color coherence (3 dims)

```python
r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
feats = [corrcoef(r,g)[0,1], corrcoef(r,b)[0,1], corrcoef(g,b)[0,1]]
```
Pearson correlation coefficient between each pair of flattened color
channels of the **original RGB** array (post-resize, pre-grayscale),
3 values: (R,G), (R,B), (G,B).

### Concatenation

`feature_vector = concatenate([fft_feats(32), dct_feats(8), color_feats(3)])`
→ **43-dim** total, in that exact order (`f0..f31` = FFT bins, `f32..f39` =
DCT stats, `f40..f42` = color coherence).

## Classifier hyperparameters (all probes, canonical going forward)

```python
StandardScaler()
LogisticRegression(max_iter=2000, C=0.1, random_state=42)
```
Trained via `clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])`
where `train_mask = ~val_mask` against the canonical eval split (see
`SPLITS.md`). This exact `(C=0.1, random_state=42, StandardScaler)`
configuration is shared by the FFT probe (`extractors/extract_fft.py` via
`experiments/e0_fft_variant.py`), SigLIP2 (`train_siglip2_classifiers.py`),
DINOv2 (`train_dinov2_classifiers.py`), and — after this Phase 0 fix — CLIP
(`pipeline/retrain_clip.py`).

## Other modules in this directory

- `splits.py` — canonical train/eval split accessors
  (`get_canonical_eval_paths`, `get_train_mask`, `get_eval_mask`) and
  `compute_split_manifest()` (writes `results/splits_manifest.csv`). See
  `SPLITS.md` for the full narrative and real counts.
- `dedup.py` — exact-file (sha256) and path-level set-overlap checks across
  benchmarks. Writes `results/image_overlap_report.csv`.
- `retrain_clip.py` — clean CLIP probe retrain for GenBuster only (fixes the
  leakage described in `SPLITS.md`).
- `verifiers.py` — consolidated `call_gpt55()` (write-only; do not call the
  live API without an `OPENAI_API_KEY`) and `call_qwen()` /
  `load_qwen_model()` (local GPU, safe to execute). Includes an offline
  parsing self-test (`python verifiers.py`).
