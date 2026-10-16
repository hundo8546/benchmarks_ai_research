#!/usr/bin/env python3
"""
routing/run_pipeline.py — thin CLI orchestrator for the "upstream" data-generation
pipeline (index building -> feature extraction -> probe training -> verifier
calls -> audit).

This script does NOT reimplement feature extraction, probe training, or
verifier calling from scratch. Every subcommand dispatches to an existing,
already-authoritative script or function listed in routing/README.md, either
by importing a function directly, or (for scripts that are a bare
`if __name__ == "__main__"`-free top-level script with no importable main())
by running that script's file in-process via runpy, exactly as `python
script.py ...` would. See each subcommand's implementation for exactly which
existing file it calls.

Subcommands:
  --build-indices [--datasets ...]
  --split
  --extract-features {fft,clip,siglip2,dinov2,cnnspot,univfd} [--datasets ...]
  --train-probes {fft,clip,siglip2,dinov2} [--datasets ...]
  --run-verifier {qwen,gpt55} --subset {disagreement,full,custom} [--datasets ...]
  --audit

Run `python run_pipeline.py <subcommand> --help` for per-subcommand options.
"""
import argparse
import os
import runpy
import sys

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROUTING_BASE)

ALL_DATASETS = ["GenBuster", "SD14", "BigGAN"]


def _run_script(rel_path, argv=None):
    """Execute an existing script file in-process, exactly as `python
    rel_path argv...` would (module-level code and all), without needing to
    refactor every target script into an importable function. Used for
    scripts that have no main()/if __name__ guard (e.g. generate_clip_
    features*.py, generate_cnnspot_features*.py, run_univfd.py) or whose
    argparse-based main() is simplest to invoke exactly as the CLI would."""
    path = os.path.join(ROUTING_BASE, rel_path)
    old_argv = sys.argv
    try:
        sys.argv = [path] + list(argv or [])
        runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = old_argv


# ─────────────────────────────────────────────────────────────────────────
# --build-indices
# ─────────────────────────────────────────────────────────────────────────
INDEX_BUILDERS = {
    # name -> (module import path, callable) for scripts refactored into
    # importable main()s during this consolidation pass.
    "SD14": "build_sd14_index",
    "BigGAN": "build_biggan_index",
    "GenBuster": "build_genbuster_index",
    # AIGIBench (SDXL/FLUX) already has its own main()+argparse from a prior
    # Phase-0 fix; owned by that agent, left untouched, called via runpy so
    # its own --n_samples flag still works unmodified.
    "AIGIBench": "build_aigibench_index.py",
}


def cmd_build_indices(args):
    for ds in args.datasets:
        if ds not in INDEX_BUILDERS:
            print(f"  [build-indices] unknown dataset {ds!r}, skipping "
                  f"(choices: {list(INDEX_BUILDERS)})")
            continue
        print(f"\n=== build-indices: {ds} ===")
        if ds == "AIGIBench":
            # Deliberately NOT invoked by default verification runs -- the
            # task that owns this says sdxl_index.csv/flux_index.csv are
            # "deliberately not yet regenerated, deferred to a later
            # paid-API phase." Wired here for completeness/parity, but a
            # human should opt in explicitly via --datasets AIGIBench.
            _run_script(INDEX_BUILDERS[ds], argv=["--n_samples", str(args.n_samples)])
        else:
            mod = __import__(INDEX_BUILDERS[ds])
            mod.main()


# ─────────────────────────────────────────────────────────────────────────
# --split
# ─────────────────────────────────────────────────────────────────────────
def cmd_split(args):
    from pipeline.splits import compute_split_manifest
    manifest = compute_split_manifest(write=True)
    print("\n" + manifest.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# --extract-features
# ─────────────────────────────────────────────────────────────────────────
# backbone -> how to dispatch per dataset. "argparse_dataset" scripts take a
# single --dataset flag (extractors/*.py); "per_dataset_script" scripts are
# one hardcoded script per dataset with no args; "all_in_one" scripts loop
# over every dataset themselves and ignore --datasets (flagged as such).
EXTRACT_DISPATCH = {
    "fft":     {"kind": "argparse_dataset", "script": "extractors/extract_fft.py"},
    "siglip2": {"kind": "argparse_dataset", "script": "extractors/extract_siglip2.py"},
    "dinov2":  {"kind": "argparse_dataset", "script": "extractors/extract_dinov2.py"},
    "clip": {
        "kind": "per_dataset_script",
        "script": {
            "GenBuster": "generate_clip_features.py",
            "SD14": "generate_clip_features_sd14.py",
            "BigGAN": "generate_clip_features_biggan.py",
        },
    },
    "cnnspot": {
        "kind": "per_dataset_script",
        "script": {
            "GenBuster": "generate_cnnspot_features.py",
            "SD14": "generate_cnnspot_features_sd14.py",
            "BigGAN": "generate_cnnspot_features_biggan.py",
        },
    },
    "univfd": {"kind": "all_in_one", "script": "run_univfd.py"},
}


def cmd_extract_features(args):
    spec = EXTRACT_DISPATCH[args.backbone]
    if spec["kind"] == "argparse_dataset":
        for ds in args.datasets:
            print(f"\n=== extract-features {args.backbone}: {ds} ===")
            _run_script(spec["script"], argv=["--dataset", ds])
    elif spec["kind"] == "per_dataset_script":
        for ds in args.datasets:
            script = spec["script"].get(ds)
            if script is None:
                print(f"  [extract-features {args.backbone}] no script for {ds!r}, skipping")
                continue
            print(f"\n=== extract-features {args.backbone}: {ds} ===")
            _run_script(script)
    elif spec["kind"] == "all_in_one":
        print(f"\n=== extract-features {args.backbone}: runs ALL datasets "
              f"internally (--datasets {args.datasets} ignored by {spec['script']}) ===")
        _run_script(spec["script"])


# ─────────────────────────────────────────────────────────────────────────
# --train-probes
# ─────────────────────────────────────────────────────────────────────────
def cmd_train_probes(args):
    if args.backbone == "fft":
        # IMPORTANT: this calls experiments/e0_fft_variant.py:get_fft_preds(),
        # the confirmed LIVE producer/loader of {prefix}fft_probe.pkl (trains
        # if missing, else loads cached + reports val accuracy), which
        # already uses the canonical split (extractors.common.load_val_paths
        # against the real {prefix}bandit_val_paths.npy -- functionally
        # identical to pipeline.splits for GenBuster/SD14/BigGAN, since that
        # npy file exists for all three and load_val_paths only falls back to
        # its own 30% split when it's missing).
        #
        # FLAGGED FOR HUMAN REVIEW (per this consolidation task's explicit
        # instruction, not silently resolved): extractors/train_probe.py is
        # a SEPARATE, generic per-feature-fraction trainer used by the E1/E6
        # limited-data ablations. It defines its OWN independent 70/30
        # train_test_split(stratify=y, random_state=42), UNRELATED to
        # pipeline.splits/the canonical {prefix}bandit_val_paths.npy files.
        # Tracing confirmed train_probe.py is NOT the path that produces the
        # live fft_probe.pkl used by regenerate_main_tables.py /
        # cross_family_deployment.py / baseline_comparison.py -- so this is
        # NOT the same leakage-class bug pipeline/retrain_clip.py fixed for
        # CLIP. It is still worth a human's attention because the name
        # similarity ("train_probe.py" vs "train the probe") invites
        # confusion, and its outputs (`{out_prefix}_frac{frac}.pkl`) are used
        # ONLY by the E1/E6 fraction-sweep experiments, never by the main
        # in-domain FFT probe pipeline. No code change made here -- this is a
        # flag, not a fix, per the task instructions.
        sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))
        import e0_fft_variant as e0
        for ds in args.datasets:
            print(f"\n=== train-probes fft: {ds} ===")
            df_val = e0.get_fft_preds(ds)
            acc = (df_val["y_fft"] == df_val["y_true"]).mean()
            print(f"  n_val={len(df_val)}  FFT val accuracy={acc:.4f}")
    elif args.backbone == "siglip2":
        import train_siglip2_classifiers as t
        for ds in args.datasets:
            print(f"\n=== train-probes siglip2: {ds} ===")
            t.train_and_predict(ds)
    elif args.backbone == "dinov2":
        import train_dinov2_classifiers as t
        for ds in args.datasets:
            print(f"\n=== train-probes dinov2: {ds} ===")
            t.train_and_predict(ds)
    elif args.backbone == "clip":
        if args.datasets != ["GenBuster"] and args.datasets != ALL_DATASETS:
            print("  [train-probes clip] NOTE: pipeline/retrain_clip.py is "
                  "GenBuster-only by design (SD14/BigGAN raw CLIP features "
                  "no longer exist on disk) -- --datasets is ignored here.")
        from pipeline.retrain_clip import retrain
        print("\n=== train-probes clip: GenBuster (only supported dataset) ===")
        retrain()


# ─────────────────────────────────────────────────────────────────────────
# --run-verifier
# ─────────────────────────────────────────────────────────────────────────
def _select_subset(ds_name, subset, custom_path=None):
    """Return a list of image/video paths to verify, per --subset."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    from extractors.common import DATASETS
    from pipeline.splits import get_canonical_eval_paths

    cfg = DATASETS[ds_name]
    prefix = cfg["prefix"]
    eval_paths = get_canonical_eval_paths(ds_name)

    if subset == "full":
        return sorted(eval_paths)

    if subset == "custom":
        if not custom_path:
            raise ValueError("--subset custom requires --custom-paths-csv <file with a 'path' column>")
        return pd.read_csv(custom_path)["path"].tolist()

    # subset == "disagreement": FFT vs SigLIP2 disagreement on the eval set,
    # same definition used throughout (main_results/baselines/gate).
    import joblib
    df_fft = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"))
    fft_cols = [c for c in df_fft.columns if c.startswith("f")]
    clf_fft, sc_fft = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl"))
    df_fft = df_fft[df_fft["path"].isin(eval_paths)].copy()
    df_fft["y_fft"] = clf_fft.predict(sc_fft.transform(df_fft[fft_cols].values.astype(np.float32)))

    df_sig = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}siglip2_features.csv"))
    sig_cols = [c for c in df_sig.columns if c.startswith("f")]
    clf_sig, sc_sig = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}siglip2_probe.pkl"))
    df_sig = df_sig[df_sig["path"].isin(eval_paths)].copy()
    df_sig["y_siglip2"] = clf_sig.predict(sc_sig.transform(df_sig[sig_cols].values.astype(np.float32)))

    merged = df_fft[["path", "y_fft"]].merge(df_sig[["path", "y_siglip2"]], on="path", how="inner")
    disagree = merged[merged["y_fft"] != merged["y_siglip2"]]
    return disagree["path"].tolist()


def cmd_run_verifier(args):
    for ds in args.datasets:
        print(f"\n=== run-verifier {args.verifier} / subset={args.subset}: {ds} ===")
        paths = _select_subset(ds, args.subset, args.custom_paths_csv)
        print(f"  Selected {len(paths)} paths to verify.")

        if args.verifier == "qwen":
            print("  NOTE: this wires up pipeline.verifiers.call_qwen()/"
                  "load_qwen_model() per-row; not executed by default here "
                  "(GPU model load) -- pass --execute to actually run it.")
            if not args.execute:
                continue
            from pipeline.verifiers import load_qwen_model, call_qwen
            processor, model, real_id, fake_id = load_qwen_model()
            from PIL import Image
            rows = []
            for p in paths:
                img = Image.open(p).convert("RGB") if not p.lower().endswith((".mp4",)) else None
                if img is None:
                    from decord import VideoReader, cpu
                    vr = VideoReader(p, ctx=cpu(0))
                    img = Image.fromarray(vr[0].asnumpy())
                resp = call_qwen(model, processor, img, real_id, fake_id)
                rows.append({"path": p, "y_qwen": resp.y_score, "p_fake": resp.p_fake})
            import pandas as pd
            pd.DataFrame(rows).to_csv(args.out or f"pipeline_qwen_{ds}_{args.subset}.csv", index=False)

        elif args.verifier == "gpt55":
            print("  NO OPENAI_API_KEY in this environment -- GPT-5.5 is NEVER "
                  "called live from this subcommand. Structure-only dry run:")
            from pipeline.verifiers import call_gpt55, VERIFIER_PROMPT
            print(f"    model-call signature: call_gpt55(client, model, b64_image, "
                  f"prompt=VERIFIER_PROMPT, ...) -- would be invoked once per "
                  f"selected path ({len(paths)} rows) if a client were supplied.")
            print(f"    prompt: {VERIFIER_PROMPT!r}")
            if paths:
                print(f"    example row that WOULD be sent: {paths[0]}")


# ─────────────────────────────────────────────────────────────────────────
# --audit
# ─────────────────────────────────────────────────────────────────────────
def cmd_audit(args):
    import numpy as np
    import pandas as pd
    from pipeline.dedup import compute_overlap_report
    from pipeline.splits import get_canonical_eval_paths, CANONICAL_DATASETS

    print("=== Overlap / duplicate check (pipeline/dedup.py) ===")
    overlap = compute_overlap_report(write=True)
    print(overlap.to_string(index=False))

    print("\n=== Coverage check: fraction of canonical eval set with a "
          "prediction on disk, per dataset x detector ===")
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    from extractors.common import DATASETS

    PRED_FILES = {
        "cnn": "bandit_dataset.csv", "clip": "clip_preds.csv",
        "siglip2": "siglip2_preds.csv", "dinov2": "dinov2_preds.csv",
        "univfd": "univfd_preds.csv", "qwen": "qwen_preds.csv",
    }
    rows = []
    for ds_name in CANONICAL_DATASETS:
        prefix = DATASETS[ds_name]["prefix"]
        eval_paths = get_canonical_eval_paths(ds_name)
        n_expected = len(eval_paths)
        for detector, fname in PRED_FILES.items():
            fpath = os.path.join(ROUTING_BASE, f"{prefix}{fname}")
            if not os.path.exists(fpath):
                rows.append({"dataset": ds_name, "detector": detector,
                             "n_expected": n_expected, "n_found": 0,
                             "coverage_pct": 0.0, "status": "MISSING_FILE"})
                continue
            df = pd.read_csv(fpath, usecols=lambda c: c == "path")
            n_found = int(df["path"].isin(eval_paths).sum())
            pct = 100.0 * n_found / n_expected if n_expected else float("nan")
            rows.append({"dataset": ds_name, "detector": detector,
                         "n_expected": n_expected, "n_found": n_found,
                         "coverage_pct": pct,
                         "status": "OK" if pct >= 99.9 else "PARTIAL"})

    coverage = pd.DataFrame(rows)
    print(coverage.to_string(index=False))

    out_path = os.path.join(ROUTING_BASE, "results", "pipeline_audit_report.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined = pd.concat([
        overlap.assign(check_type="overlap"),
        coverage.assign(check_type="coverage"),
    ], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


# ─────────────────────────────────────────────────────────────────────────
# argparse wiring
# ─────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Thin CLI orchestrator for the routing/ data-generation "
                     "pipeline -- dispatches to existing authoritative "
                     "scripts (see routing/README.md), never reimplements "
                     "them.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("build-indices", help="Build index CSVs for one or more datasets.")
    sp.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                     choices=list(INDEX_BUILDERS), help="Datasets to build indices for.")
    sp.add_argument("--n_samples", type=int, default=2000,
                     help="Per-class sample count, AIGIBench only.")
    sp.set_defaults(func=cmd_build_indices)

    sp = sub.add_parser("split", help="(Re)compute the canonical split manifest "
                                        "(results/splits_manifest.csv) via pipeline/splits.py.")
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser("extract-features", help="Extract features for a backbone.")
    sp.add_argument("--backbone", required=True, choices=list(EXTRACT_DISPATCH))
    sp.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    sp.set_defaults(func=cmd_extract_features)

    sp = sub.add_parser("train-probes", help="Train (or refresh) a probe for a backbone.")
    sp.add_argument("--backbone", required=True, choices=["fft", "clip", "siglip2", "dinov2"])
    sp.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    sp.set_defaults(func=cmd_train_probes)

    sp = sub.add_parser("run-verifier", help="Run a verifier (Qwen or GPT-5.5) over a subset.")
    sp.add_argument("--verifier", required=True, choices=["qwen", "gpt55"])
    sp.add_argument("--subset", required=True, choices=["disagreement", "full", "custom"])
    sp.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    sp.add_argument("--custom-paths-csv", default=None,
                     help="CSV with a 'path' column, required if --subset custom.")
    sp.add_argument("--execute", action="store_true",
                     help="Actually call the verifier (Qwen: loads the local GPU model and "
                          "runs inference. GPT-5.5: ALWAYS a structure-only dry run in this "
                          "environment -- no OPENAI_API_KEY is configured, so the live API is "
                          "never called regardless of this flag).")
    sp.add_argument("--out", default=None, help="Output CSV path (Qwen only).")
    sp.set_defaults(func=cmd_run_verifier)

    sp = sub.add_parser("audit", help="Run the image-overlap + per-detector coverage audit.")
    sp.set_defaults(func=cmd_audit)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
