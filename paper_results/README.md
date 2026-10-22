# paper_results

`paper_results.xlsx` holds every number printed in the paper (`claude_context_notes/paper_updates2.tex`), one sheet per table named by its LaTeX label, plus `inline_numbers`, `figures`, `coverage`, `gate_per_config`, `provenance` and `audit`.

`paper_results_long.csv` has one row per value, so `git diff` shows exactly which numbers changed between runs.

The `audit` sheet renders each table's LaTeX rows from its CSV and checks that they appear verbatim in the manuscript. The `README` sheet's `audit_mismatches` must be 0 before submission.

Regenerate: `python routing/generate_paper_results.py` (runs this as its last step), or `python routing/build_results_bundle.py` to rebuild from the existing CSVs. Neither calls a model or an API.
