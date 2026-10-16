"""
routing.pipeline.tables — one importable module per paper-table family.

Each module exposes one or more `compute_<table>()` functions returning a
pandas.DataFrame, with NO heavy (torch/cv2/decord/GPU) import at module load
time. Where a table's inputs require pipeline.splits (which transitively
imports extractors.common, which imports torch/cv2/decord), that import is
deferred to inside the compute function, never at module top level -- this
keeps `import pipeline.tables.<x>` or `python generate_paper_results.py
--help` fast and GPU-free, while still letting the actual computation use the
real, canonical split logic (not a hand-rolled duplicate).

generate_paper_results.py imports these modules and calls their compute_*()
functions in a loop, writing each result to results/<label>.csv. Follow-on
work (confusion matrices, cost accounting, new gate rules, etc.) should add
new sibling modules here, each with its own compute_*() function, rather than
growing generate_paper_results.py itself.
"""
