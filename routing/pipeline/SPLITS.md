# The actual train/eval split scheme (ground truth, Phase 0 remediation)

This document replaces the fictional "50% train / 20% probe-val / 50% routing
pool / 40% held-out eval" scheme described in
`claude_context_notes/paper_updates2.tex` (~lines 498-504). That text does not
match the code and does not reproduce the paper's own reported eval sizes
(1000 / 2000 / 2000). The .tex is **not edited in this task** — this file is
ground truth for whoever rewrites the Methods section next.

## The real scheme, in one sentence

For each of GenBuster / SD14 / BigGAN there is exactly one fixed eval-set
file, `{prefix}bandit_val_paths.npy` (written once by
`routing/expand_val_splits.py`, stratified 50/50 real/fake, seed 42). Every
probe (FFT, SigLIP2, DINOv2, and — after this fix — CLIP) is trained on
*everything else* in the dataset: `train_mask = ~val_mask`. There is no
separate probe-validation holdout and no separate "routing pool" — the single
eval set measures both probe accuracy and cascade/routing accuracy.

## Per-dataset real counts (measured, not estimated)

Computed by `routing/pipeline/splits.py::compute_split_manifest()`, written
to `routing/results/splits_manifest.csv`:

| dataset   | total_pool | n_train | n_eval | n_eval_real | n_eval_fake |
|-----------|-----------:|--------:|-------:|------------:|------------:|
| GenBuster |       2000 |    1000 |   1000 |         500 |         500 |
| SD14      |      12000 |   10000 |   2000 |        1000 |        1000 |
| BigGAN    |      12000 |   10000 |   2000 |        1000 |        1000 |

Notes:
- GenBuster's *total* raw pool is only 2000 samples (its native size — there
  is no larger upstream pool to draw from), so n_train=1000 is a real
  ceiling, not a choice. This is why GenBuster's eval set (1000) is smaller
  than SD14/BigGAN's (2000) — it is exactly half of everything there is.
- SD14 and BigGAN each have the same 12,000-image GenImage pool (6000 real +
  6000 fake per family), fully feature-extracted; `expand_val_splits.py`
  widened their eval sets from an old small split to 2000 (still well under
  half the pool, so the probes still train on 10,000 of the 12,000 samples).
- The eval split is stratified exactly 50/50 real/fake by construction
  (`expand_val_splits.py`: `n_pos_val = n_val // 2`, `n_neg_val = n_val -
  n_pos_val`), not by incidental class balance.
- "n_train" above is `total_pool - n_eval`, i.e. it assumes every pool row
  that isn't in the eval set is used for training. This is what
  `train_mask = ~val_mask` actually does; it is not literally "50% train".

## How any script gets this split (the one true way)

```python
from extractors.common import DATASETS, load_val_paths
# or, equivalently and preferred going forward:
from pipeline.splits import get_canonical_eval_paths, get_train_mask

cfg = DATASETS[ds_name]
val_paths = load_val_paths(cfg, df["path"].tolist())   # reads {prefix}bandit_val_paths.npy
val_mask = df["path"].isin(val_paths)
train_mask = ~val_mask
```

`pipeline/splits.py` is a clean wrapper around exactly this — it does not
regenerate `*_bandit_val_paths.npy`, it only reads it. Any new probe-training
script should import `get_canonical_eval_paths` / `get_train_mask` from
`pipeline.splits` rather than re-deriving a split (this is what caused the
CLIP leakage bug in the first place: `train_clip_classifier.py` built its own
independent `train_test_split` + `GroupShuffleSplit` scheme instead of using
the shared eval file).

## What was fixed, what was not (Phase 0 scope)

- **Fixed**: CLIP's probe for **GenBuster only** now uses this scheme
  (`routing/pipeline/retrain_clip.py`, overwriting `clip_classifier.pkl` /
  `clip_preds.csv`; leaky originals backed up with `.leaky_backup` suffix).
- **Not fixed / unverified**: SD14 and BigGAN's CLIP probes. Their raw CLIP
  feature files (`sd14_clip_features.csv`, `biggan_clip_features.csv`) no
  longer exist on disk, so whether their saved `*_clip_preds.csv` /
  `*_clip_val_idx.npy` suffer the same leakage as GenBuster's CLIP could not
  be verified in this pass, and regenerating the missing feature files from
  raw images was out of scope for this task. Treat SD14/BigGAN CLIP numbers
  as possibly-affected until the feature files are regenerated and the probe
  retrained the same way.
- **Not changed**: the existing `*_bandit_val_paths.npy` files themselves
  (GenBuster=1000, SD14=2000, BigGAN=2000) — these were already correct and
  are left untouched, per the Phase 0 mandate.

## SDXL / FLUX: real-image sourcing is a *different* mechanism, not a split

SDXL and FLUX are not probed in-domain — they are deployment-shift eval sets
built by `routing/build_aigibench_index.py` to test SD14-trained ("stale")
probes zero-shot (see `routing/experiments/e8_modern_diffusion.py`). They are
not derived from `expand_val_splits.py`'s GenImage split logic at all.
Instead:

- **Fake images**: sampled from the AIGIBench SDXL/FLUX generated-image
  directories (`data/aigibench/sdxl/1_fake`, `data/aigibench/flux/1_fake`),
  2000 per family, seed 42.
- **Real images**: sampled from SD1.4's own real-image pool
  (`data/genimage/sd_1_4/imagenet_ai_0419_sdv4/val/nature`, the full
  ~6000-image pool used as "real" for SD14 training/eval), 2000 per family,
  seed 42, via `random.sample(real_imgs, n_samples)`.

**Bug (Phase 0 item C, fixed in code, not yet re-run)**: that real-image
sample was drawn independently of SD14's own eval split
(`sd14_bandit_val_paths.npy`), so ~1000 of the real images used to measure
SD14 in-domain accuracy were landing in the "real" pool sampled for SDXL/FLUX
deployment-shift accuracy too — i.e. the same real photo could be scored once
as part of "SD14 in-domain" and again as part of "SDXL/FLUX deployment
shift", inflating the appearance of independence between those two numbers.
`routing/build_aigibench_index.py` has been patched to exclude SD14's
eval-set real images from the `random.sample()` pool before sampling (same
seed=42, same n_samples=2000, same output format) — see
`routing/pipeline/dedup.py` for the measured overlap and a dry-run
confirmation that the exclusion works. **The real `sdxl_index.csv` /
`flux_index.csv` have NOT been regenerated** — doing so would require
re-extracting FFT/CLIP/SigLIP2/DINOv2 features and new GPT-5.5 calls for
swapped-in images, which needs API budget approval (deferred; see main
report, item C).
