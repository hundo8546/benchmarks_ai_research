#!/usr/bin/env python3
"""
routing/generate_paper_results.py — CPU-only "downstream" results generator.

Reads only already-computed CSVs/pkls from disk (features, predictions,
probes) and writes one CSV per paper table to routing/results/, named to
match its LaTeX \\label{}. It never calls a model or an API -- all numbers
come from predictions already on disk (plus cheap in-process LogisticRegression
refits on cached features for a few of the tables, matching what the ported
source scripts already did).

CPU-only constraint (same one regenerate_main_tables.py already imposed on
itself -- see its docstring): NO torch/cv2/decord/GPU import at module load
time. This script's own top-level imports are only os/sys/datetime/
subprocess/traceback/pandas. The actual table logic lives in
pipeline/tables/*.py, each of which also keeps its OWN top level free of
heavy imports and only imports pipeline.splits / extractors.common (which
pull in torch/cv2/decord as an unwanted side effect of needing DATASETS /
np.load) lazily, inside the function that does the real computation. Import
this module or run `--help` and nothing heavy loads; run it for real and the
heavy imports happen exactly where the comments in pipeline/tables/*.py say
they do.

Module boundaries (for follow-on work -- confusion matrices, cost
accounting, a case-mapping table, new gate rules, a confidence-routing
baseline, compression/resolution ablations, etc.): each paper table (or
closely related family of tables sharing one computation, like main_results/
cheap_path/escalation) lives in its own file under pipeline/tables/, each
exposing one or more `compute_<table>() -> pd.DataFrame` functions:

  pipeline/tables/main_results.py   tab_main_results, tab_cheap_path, tab_escalation
  pipeline/tables/baselines.py      tab_baselines, tab_verifier_comparison, tab_e8_results,
                                     tab_cross_family_deployment, tab_limited_data, tab_biggan_shift
  pipeline/tables/gate.py           tab_sensitivity, tab_threeway
  pipeline/tables/transfer.py       tab_transfer
  pipeline/tables/per_generator.py  tab_per_generator
  pipeline/tables/cheap_trigger.py  tab_cheap_trigger (NEWLY IMPLEMENTED, no prior live source -- see its docstring)
  pipeline/tables/confidence_routing.py  tab_confidence_routing
  pipeline/tables/gate_holdout.py   tab_gate_holdout, tab_agreement_path_audit
  pipeline/tables/e8_cascade.py     tab_e8_cascade, tab_cost_accuracy_tradeoff, tab_real_image_sources
  pipeline/tables/e8_sizematched.py tab_e8_cascade_sizematched (original vs. size-matched SDXL/FLUX)
  pipeline/tables/robustness.py     tab_robustness (reads e7/e9 robustness outputs)
  pipeline/tables/cost_accounting.py tab_cost_accounting
  pipeline/tables/case_mapping.py   tab_case_mapping (run last: reads other tab_*.csv)
  pipeline/tables/baselines.py also writes tab_limited_data_cascade and
                                     tab_cross_family_deployment_cascade (the printed table columns)

To add a new table: add a new sibling module under pipeline/tables/ exposing
its own compute_<x>() -> pd.DataFrame, then add one entry to the TABLES list
below (or, for a brand-new independent table family, just call it directly
in main() the same way the grouped families are called). Nothing else in
this file needs to change.
"""
import datetime
import os
import subprocess
import sys
import traceback

import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROUTING_BASE, "results")
sys.path.insert(0, ROUTING_BASE)

from pipeline.tables import (main_results, per_generator, cheap_trigger, baselines, gate, transfer,
                             confidence_routing, gate_holdout, e8_cascade, e8_sizematched)


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROUTING_BASE,
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else f"ERROR:{out.stderr.strip()}"
    except Exception as e:
        return f"ERROR:{e}"


def _write(label, df, log):
    out_path = os.path.join(RESULTS_DIR, f"{label}.csv")
    df.to_csv(out_path, index=False)
    log.append(f"[OK] {label}.csv  n_rows={len(df)}  -> {out_path}")
    print(f"  [OK] {label}: {len(df)} rows -> {out_path}")


def _fail(label, exc, log):
    msg = f"[FAIL] {label}: {exc!r}"
    log.append(msg)
    log.append("".join(traceback.format_exception(exc)).strip())
    print(f"  {msg}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = []
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log.append(f"generate_paper_results.py run started {started} UTC")
    log.append(f"git commit: {_git_commit()}")
    log.append("")

    # ---- main_results family (also fixes the GenBuster-split bug: now goes
    # through pipeline.splits.get_canonical_eval_paths, not a fallback split) --
    print("=== main_results / cheap_path / escalation ===")
    try:
        df_main, df_cheap, df_esc, warn = main_results.compute_all_with_warnings()
        for w in warn:
            log.append(f"[WARN] {w}")
            print(f"  [WARN] {w}")
        _write("tab_main_results", df_main, log)
        _write("tab_cheap_path", df_cheap, log)
        _write("tab_escalation", df_esc, log)
        genbuster_fft_acc = df_main[(df_main.dataset == "GenBuster") & (df_main.method == "FFT")]["accuracy"].iloc[0]
        log.append(f"[CHECK] GenBuster FFT accuracy = {genbuster_fft_acc:.4f} "
                    f"(expect 0.727, matching the paper; NOT 0.7413, the known "
                    f"split-fallback-bug value)")
        print(f"  [CHECK] GenBuster FFT accuracy = {genbuster_fft_acc:.4f} (expect 0.727)")
    except Exception as e:
        _fail("tab_main_results/tab_cheap_path/tab_escalation", e, log)

    # ---- baselines family: compute the raw summary ONCE, slice 6 ways -------
    print("\n=== baselines family (verifier_comparison / e8_results / "
          "cross_family_deployment / limited_data / biggan_shift) ===")
    try:
        df_base_raw, bc = baselines.compute_baselines_raw()
        _write("tab_baselines", df_base_raw, log)
        for family in baselines.TABLE_FAMILIES:
            sub = df_base_raw[df_base_raw["table"] == family].reset_index(drop=True)
            _write(f"tab_{family}", sub, log)
        df_casc = pd.DataFrame(bc.CASCADE_ROWS)
        for family in ("limited_data", "cross_family_deployment"):
            sub = df_casc[df_casc["table"] == family].dropna(axis=1, how="all").reset_index(drop=True)
            _write(f"tab_{family}_cascade", sub, log)
    except Exception as e:
        _fail("tab_baselines (and its per-family views)", e, log)

    # ---- gate family: compute the raw sweep ONCE, slice 2 ways --------------
    print("\n=== gate family (sensitivity / threeway) ===")
    try:
        df_gate_raw = gate.compute_sweep_raw()
        _write("tab_sensitivity", df_gate_raw[
            ["table", "config", "verifier", "n_disagree", "calib_N",
             "binary_agreement_rate", "oracle_binary_escalate"]], log)
        _write("tab_threeway", df_gate_raw[
            ["table", "config", "verifier", "n_disagree", "calib_N",
             "three_way_agreement_rate", "oracle_three_way_choice"]], log)
    except Exception as e:
        _fail("tab_sensitivity/tab_threeway", e, log)

    # ---- transfer -------------------------------------------------------------
    print("\n=== transfer ===")
    try:
        _write("tab_transfer", transfer.compute_transfer(), log)
        _write("tab_transfer_reconciliation", transfer.compute_transfer_reconciliation(), log)
    except Exception as e:
        _fail("tab_transfer", e, log)

    # ---- cheap_trigger (newly implemented -- see pipeline/tables/cheap_trigger.py) --
    print("\n=== cheap_trigger (newly implemented, no prior live source) ===")
    try:
        _write("tab_cheap_trigger", cheap_trigger.compute_cheap_trigger(), log)
    except Exception as e:
        _fail("tab_cheap_trigger", e, log)

    print("\n=== e8_cascade / cost_accuracy_tradeoff / real_image_sources ===")
    for label, fn in [("tab_e8_cascade", e8_cascade.compute_e8_cascade),
                      ("tab_cost_accuracy_tradeoff", e8_cascade.compute_cost_accuracy_tradeoff),
                      ("tab_real_image_sources", e8_cascade.compute_real_image_sources),
                      ("tab_positive_cell_audit", e8_cascade.compute_positive_cell_audit),
                      ("tab_e8_cascade_sizematched", e8_sizematched.compute_e8_sizematched)]:
        try:
            _write(label, fn(), log)
        except Exception as e:
            _fail(label, e, log)

    print("\n=== gate_holdout / agreement_path_audit ===")
    try:
        _write("tab_gate_holdout", gate_holdout.compute_gate_holdout_validation(), log)
        _write("tab_agreement_path_audit", gate_holdout.compute_agreement_path_audit(), log)
    except Exception as e:
        _fail("tab_gate_holdout/tab_agreement_path_audit", e, log)

    print("\n=== confidence_routing ===")
    try:
        _write("tab_confidence_routing", confidence_routing.compute_confidence_routing_comparison(), log)
    except Exception as e:
        _fail("tab_confidence_routing", e, log)

    print("\n=== per_generator ===")
    try:
        _write("tab_per_generator", per_generator.compute_per_generator(), log)
    except Exception as e:
        _fail("tab_per_generator", e, log)

    print("\n=== cost_accounting / case_mapping (reads the tables above) ===")
    for label, mod_name, fn_name in [("tab_robustness", "robustness", "compute_robustness"),
                                     ("tab_cost_accounting", "cost_accounting", "compute_cost_accounting"),
                                     ("tab_biggan_resolution", "biggan_resolution", "compute_biggan_resolution"),
                                     ("tab_case_mapping", "case_mapping", "compute_case_mapping")]:
        try:
            mod = __import__(f"pipeline.tables.{mod_name}", fromlist=[fn_name])
            _write(label, getattr(mod, fn_name)(), log)
        except Exception as e:
            _fail(label, e, log)

    print("\n=== figures ===")
    for script in ("plot_main_results_figures.py",
                   "plot_fig_a_semantic_degradation.py", "plot_fig_b_cascade_gain_scatter.py"):
        out = subprocess.run([sys.executable, os.path.join(ROUTING_BASE, "experiments", script)],
                             capture_output=True, text=True)
        status = "OK" if out.returncode == 0 else f"FAIL: {out.stderr.strip()[-500:]}"
        log.append(f"[{'OK' if out.returncode == 0 else 'FAIL'}] figure script {script}")
        print(f"  [{status}] {script}")

    print("\n=== paper_results bundle + audit ===")
    try:
        import build_results_bundle
        n_bad, _, _ = build_results_bundle.build()
        log.append(f"[{'OK' if n_bad == 0 else 'AUDIT'}] paper_results bundle: {n_bad} audit mismatches")
        print(f"  [{'OK' if n_bad == 0 else 'AUDIT'}] paper_results/ written, {n_bad} audit mismatches")
    except Exception as e:
        _fail("paper_results bundle", e, log)

    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log.append("")
    log.append(f"run finished {finished} UTC")

    log_path = os.path.join(RESULTS_DIR, "pipeline_audit_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nWrote {log_path}")


if __name__ == "__main__":
    main()
