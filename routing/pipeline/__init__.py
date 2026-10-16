"""
routing.pipeline — Phase 0 data-pipeline remediation utilities.

Submodules:
    splits.py     Canonical train/eval split definitions (single source of truth).
    dedup.py       Exact-file and path-level overlap/duplicate detection.
    retrain_clip.py  Clean CLIP probe retraining using the canonical split.
    verifiers.py   Consolidated GPT-5.5 / Qwen calling code (write-only for GPT-5.5).
"""
