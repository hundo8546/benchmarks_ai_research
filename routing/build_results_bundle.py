#!/usr/bin/env python3
"""
routing/build_results_bundle.py — one file holding every number the paper prints.

CPU-only. Reads routing/results/*.csv and the manuscript; never calls a model or an API.
Writes (outside routing/, so it is not an LFS object and is readable on GitHub):
  paper_results/paper_results.xlsx     README, one sheet per paper table, inline_numbers,
                                       figures, coverage, gate_per_config, provenance, audit
  paper_results/paper_results_long.csv one row per value: table,row_key,column,value,source_csv,source_column
  paper_results/README.md

The audit sheet renders each table's LaTeX rows from its CSV (half-up rounding, as printed)
and checks that every row appears verbatim (whitespace-normalised) in the manuscript, and
recomputes each headline inline number. It must show zero mismatches before submission.
Run directly, or as the last step of generate_paper_results.py.
"""
import datetime
import os
import re
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ROUTING = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROUTING)
RESULTS = os.path.join(ROUTING, "results")
OUT_DIR = os.path.join(REPO, "paper_results")
TEX = os.path.join(REPO, "claude_context_notes", "paper_updates2.tex")


# ── formatting (matches the manuscript: half-up on the float's repr) ─────────────
def q(x, places=3):
    # round(.., 10) strips float noise first, so 0.7675 - 0.713 = 0.0544999... prints as 0.055
    return str(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def sgn(x, places=3, math=True):
    v = Decimal(q(x, places))
    mark = ("$-$" if v < 0 else "$+$") if math else ("-" if v < 0 else "+")
    return mark + str(abs(v))


def pct(x, places=1):
    return q(100 * x, places) + r"\%"


def n_tex(n):
    return f"{int(n):,}".replace(",", "{,}")


def row(*cells):
    return " & ".join(str(c) for c in cells) + r" \\"


def csv(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}.csv"))


DS = {"GenBuster": "GenBuster", "SD14": "SD~1.4", "BigGAN": "BigGAN", "SDXL": "SDXL", "FLUX": "FLUX"}


# ── per-table renderers: (sheet DataFrame, expected LaTeX rows) ──────────────────
def t_main_results():
    d = csv("tab_main_results")
    order = ["FFT", "CNNSpot", "UnivFD", "CLIP", "SigLIP2", "DINOv2", "Entropy",
             "Casc.(C)", "Casc.(S)", "Casc.(D)", "Always esc."]
    rows = []
    for m in order:
        cells = []
        for ds in ["GenBuster", "SD14", "BigGAN"]:
            r = d[(d.dataset == ds) & (d.method == m)].iloc[0]
            cells += [q(r.accuracy), q(r.latency_ms, 1)]
        rows.append(row(m.replace("Casc.(", "Casc. ("), *cells))
    return d, rows, "tab_main_results"


def t_verifier_comparison():
    d = csv("tab_verifier_comparison")
    rows = [row(DS[r.config], r.verifier, q(r.verifier_acc), r.n, q(r.best_cheap_acc),
                "Yes" if r.clears_best_cheap else "No") for r in d.itertuples()]
    return d, rows, "tab_verifier_comparison"


def t_baselines():
    d = csv("tab_baselines")
    n = len(d)
    rows = [row("Random", f"{int(d.clears_random.sum())} / {n}"),
            row("Majority-class", f"{int(d.clears_majority.sum())} / {n}"),
            row("Best-cheap-detector", f"{int(d.clears_best_cheap.sum())} / {n}")]
    return d, rows, "tab_baselines"


def t_e8_results():
    d = csv("tab_e8_cascade")
    rows = []
    for sem in ["SigLIP2", "DINOv2"]:
        cells = []
        for ds in ["SDXL", "FLUX"]:
            r = d[(d.dataset == ds) & (d.semantic == sem)].iloc[0]
            cells += [f"{q(r.cascade_acc_qwen)} / {sgn(r.gain_qwen)}", f"{q(r.cascade_acc_gpt55)} / {sgn(r.gain_gpt55)}"]
        rows.append(row(sem, *cells))
    return d, rows, "tab_e8_cascade"


def t_positive_cells():
    d = csv("tab_positive_cell_audit")
    rows = [row(f"{r.dataset}/{r.semantic}", n_tex(r.n_disagree), f"{r.n_real}/{r.n_fake}", q(r.gpt55_acc),
                q(r.best_cheap_acc), f"{sgn(r.gap)} [{q(r.gap_ci_lo)}, {q(r.gap_ci_hi)}]") for r in d.itertuples()]
    return d, rows, "tab_positive_cell_audit"


def t_real_image_sources():
    d = csv("tab_real_image_sources")
    e8 = csv("tab_e8_cascade")
    name = {"original": "PASCAL VOC (original)", "coco": "COCO (independent)"}
    rows = [row(name[r.source], r.dataset, r.semantic, q(r.agreement_path_acc), q(r.cascade_acc_gpt55),
                sgn(r.gain_over_stale_semantic, math=False)) for r in d.itertuples()]
    rows += [row("ImageNet (used throughout)", r.dataset, r.semantic, "--", q(r.cascade_acc_gpt55),
                 sgn(r.gain_gpt55, math=False)) for r in e8.itertuples()]
    return d, rows, "tab_real_image_sources + tab_e8_cascade"


def t_escalation():
    d = csv("tab_escalation")
    return d, [row(r.case, r.n, q(r.disagree_esc, 2)) for r in d.itertuples()], "tab_escalation"


def t_per_generator():
    d = csv("tab_per_generator")
    rows = [row(r.generator, r.n, q(r.cnnspot_acc), q(r.fft_acc), q(r.siglip2_acc), q(r.cascade_acc))
            for r in d.itertuples()]
    return d, rows, "tab_per_generator"


def t_cheap_path():
    d = csv("tab_cheap_path")
    rows = [row(DS[r.dataset], r.n_agree, q(r.cheap_acc), r.n_disagree, q(r.esc_rate)) for r in d.itertuples()]
    return d, rows, "tab_cheap_path"


def t_cheap_trigger():
    d = csv("tab_cheap_trigger")
    rows = [row(DS[r.dataset], q(r.esc_rate), q(r.disagree_cascade_acc),
                f"{q(r.random_cascade_mean)} [{q(r.random_cascade_ci_lo)}, {q(r.random_cascade_ci_hi)}]",
                "Yes" if r.beats_random else "No") for r in d.itertuples()]
    return d, rows, "tab_cheap_trigger"


def t_limited_data():
    d = csv("tab_limited_data_cascade")
    rows = [row(r.backbone, f"{int(round(r.train_frac * 100))}\\%", q(r.sem_only_acc), q(r.cascade_acc), sgn(r.gain))
            for r in d.itertuples()]
    return d, rows, "tab_limited_data_cascade"


def t_transfer():
    d = csv("tab_transfer")
    acc = d.set_index(["train", "test"]).accuracy
    cols = ["GenBuster", "BigGAN", "SD14"]
    rows = []
    for tr in cols:
        cells = [rf"\textbf{{{q(acc[(tr, te)])}}}" if te == tr else q(acc[(tr, te)]) for te in cols]
        rows.append(row(tr, *cells))
    main = csv("tab_main_results")
    rule = [q(main[(main.dataset == ds) & (main.method == "Casc.(S)")].accuracy.iloc[0]) for ds in cols]
    rows.append(row("Disagreement rule (no training)", *rule))
    return d, rows, "tab_transfer + tab_main_results"


def t_cross_family():
    d = csv("tab_cross_family_deployment_cascade")
    rows = []
    for r in d.itertuples():
        dag = r"$^\dagger$" if r.delta_fft_ci_lo <= 0 <= r.delta_fft_ci_hi else ""
        rows.append(row(DS[r.train], DS[r.test], q(r.fft_acc), q(r.stale_siglip2_acc), q(r.cascade_acc),
                        sgn(r.delta_stale), sgn(r.delta_fft) + dag))
    rows.append(r"\multicolumn{2}{l}{Mean} & & & & " + f"{sgn(d.delta_stale.mean())} & {sgn(d.delta_fft.mean())}" + r" \\")
    return d, rows, "tab_cross_family_deployment_cascade"


def t_biggan_shift():
    d = csv("tab_biggan_shift")
    return d, [row(r.verifier, q(r.verifier_acc), q(r.best_cheap_acc)) for r in d.itertuples()], "tab_biggan_shift"


def t_biggan_resolution():
    d = csv("tab_biggan_resolution")
    rows = [row(r.verifier if r.resolution == "native" else "", r.resolution, q(r.acc), q(r.real_recall),
                q(r.fake_recall), q(r.pred_fake_rate)) for r in d.itertuples()]
    return d, rows, "tab_biggan_resolution"


def t_confidence_routing():
    d = csv("tab_confidence_routing")
    p = d[(d.rate_matching == "val_tau") & (d.fallback == "semantic")]
    ver = {"Qwen2.5-VL-7B": "Qwen-7B", "GPT-5.5": "GPT-5.5"}
    rows = []
    for r in p.itertuples():
        cfg = DS[r.dataset] + ("" if r.regime == "indomain" else "/" + r.semantic)
        sig = "" if r.diff_ci_lo <= 0 <= r.diff_ci_hi else r"$^\ast$"
        rows.append(row(cfg, ver[r.verifier], n_tex(r.n_eval),
                        f"{q(r.disagreement_rule_acc)} ({q(r.disagreement_rule_esc_rate)})",
                        f"{q(r.confidence_rule_acc)} ({q(r.confidence_rule_esc_rate)})",
                        f"{sgn(r.acc_diff_conf_minus_dis)}{sig} [{sgn(r.diff_ci_lo)}, {sgn(r.diff_ci_hi)}]"))
    return d, rows, "tab_confidence_routing"


def t_sensitivity():
    s, t = csv("tab_sensitivity"), csv("tab_threeway")
    n = s[["table", "config", "verifier"]].drop_duplicates().shape[0]
    Ns = [10, 20, 50, 100, 200]
    rows = [row("Binary gate", *[f"{int((s[s.calib_N == N].binary_agreement_rate >= .9).sum())}/{n}" for N in Ns]),
            row("Three-way selector", *[f"{int((t[t.calib_N == N].three_way_agreement_rate >= .9).sum())}/{n}" for N in Ns])]
    return s, rows, "tab_sensitivity + tab_threeway"


def t_gate_holdout():
    b = csv("tab_gate_holdout")
    e = b[~b.pool_too_small]
    S = ["N=50", "N=100", "N=200", "half"]
    n = e[["table", "config", "verifier"]].drop_duplicates().shape[0]
    agg = lambda col, f: [f(e[e.split == s][col]) for s in S]
    rows = []
    for tag, name in [("naive", "Naive (point-estimate)"), ("sig", "Significance-aware")]:
        rows.append(row(name, *[f"{v}/{n}" for v in agg(f"{tag}_test_oracle_match_rate", lambda x: int((x >= .9).sum()))]))
        rows.append(row(r"\ \ harmful-esc.\ rate", *[pct(v) for v in agg(f"{tag}_harmful_escalation_rate", lambda x: x.mean())]))
        rows.append(row(r"\ \ missed-benefit rate", *[pct(v) for v in agg(f"{tag}_missed_beneficial_escalation_rate", lambda x: x.mean())]))
        rows.append(row(r"\ \ mean regret (points)", *[q(100 * v, 2) for v in agg(f"{tag}_mean_regret", lambda x: x.mean())]))
    rows.append(row("Rules choose differently", *[pct(v) for v in agg("rule_disagreement_rate", lambda x: x.mean())]))
    p = b[(b.table == "e8_results") & (b.verifier == "GPT-5.5")]
    for c in ["SDXL/SigLIP2", "SDXL/DINOv2", "FLUX/SigLIP2", "FLUX/DINOv2"]:
        r = p[p.config == c].set_index("split").sig_escalation_rate
        rows.append(row(rf"\ \ {c}", *[pct(r[s]) for s in S]))
    return b, rows, "tab_gate_holdout"


CASE_GROUPS = [  # (label in the paper, regex on tab_case_mapping.configuration)
    ("In-domain GenBuster, SD~1.4", r"^(?:GenBuster|SD14) in-domain"),
    ("In-domain BigGAN", r"^BigGAN in-domain"),
    ("SDXL/FLUX shift, Qwen-7B", r"deployment shift.*Qwen"),
    ("SDXL/FLUX shift, GPT-5.5", r"deployment shift.*GPT-5\.5"),
    ("Cross-family deployment", r"^cross-family"),
    ("BigGAN distant shift", r"^BigGAN distant shift"),
    ("Probe-degradation sweep", r"probe-degradation"),
    ("Agreement path (GenBuster)", r"agreement-path"),
]


def t_case_mapping():
    d = csv("tab_case_mapping")
    rows, used = [], set()
    for label, pat in CASE_GROUPS:
        g = d[d.configuration.str.contains(pat, regex=True)]
        used |= set(g.index)
        c = g.case.astype(str).value_counts()
        rows.append(row(label, len(g), *[c.get(k, 0) or "--" for k in ["1", "2", "3", "Breakdown"]]))
    if len(used) != len(d):
        raise ValueError(f"case-mapping groups cover {len(used)} of {len(d)} rows")
    return d, rows, "tab_case_mapping"


def t_threeway():
    t = csv("tab_threeway")
    o = t.drop_duplicates(["table", "config", "verifier"]).oracle_three_way_choice.value_counts()
    n = int(o.sum())
    rows = [row(r"Semantic probe ($f_2$)", f"{o.get('cheap2', 0)} / {n}"),
            row(r"Verifier ($f_3$)", f"{o.get('verifier', 0)} / {n}"),
            row(r"Artifact detector ($f_1$)", f"{o.get('cheap1', 0)} / {n}")]
    return t, rows, "tab_threeway"


def t_cost():
    d = csv("tab_cost_accuracy_tradeoff")
    rows = [row(r.dataset, r.semantic, q(r.cheap_only_acc), f"{q(r.cascade_acc)} (\\${q(r.cascade_cost_usd, 2)})",
                f"{q(r.uniform_acc)} (\\${q(r.uniform_cost_usd, 2)})", q(r.gap_to_uniform),
                q(100 * r.cascade_cost_frac_of_uniform, 0) + r"\%") for r in d.itertuples()]
    return d, rows, "tab_cost_accuracy_tradeoff"


def t_robustness():
    d = csv("tab_robustness")
    rows = [row(DS[r["dataset"]], r["detector"], *[q(r[c]) for c in d.columns[2:]]) for _, r in d.iterrows()]
    return d, rows, "tab_robustness"


TABLES = [  # paper order; sheet name = LaTeX label
    ("main_results", t_main_results), ("verifier_comparison", t_verifier_comparison),
    ("baselines", t_baselines), ("e8_results", t_e8_results), ("positive_cells", t_positive_cells),
    ("real_image_sources", t_real_image_sources), ("escalation", t_escalation),
    ("per_generator", t_per_generator), ("confidence_routing", t_confidence_routing),
    ("case_mapping", t_case_mapping),
    ("cheap_path", t_cheap_path), ("cheap_trigger", t_cheap_trigger), ("limited_data", t_limited_data),
    ("transfer", t_transfer), ("cross_family_deployment", t_cross_family), ("biggan_shift", t_biggan_shift),
    ("biggan_resolution", t_biggan_resolution),
    ("robustness", t_robustness), ("sensitivity", t_sensitivity), ("gate_holdout", t_gate_holdout),
    ("threeway", t_threeway), ("cost_accuracy_tradeoff", t_cost),
]


# ── headline inline numbers: (location, text quoted in the .tex, source, recompute) ──
def inline_checks():
    e8, cost = csv("tab_e8_cascade"), csv("tab_cost_accuracy_tradeoff")
    gh = csv("tab_gate_holdout")
    gh = gh[~gh.pool_too_small]
    cr = csv("tab_confidence_routing")
    base = csv("tab_baselines")
    n_el = gh[["table", "config", "verifier"]].drop_duplicates().shape[0]
    cnt = lambda sp, tag: int((gh[gh.split == sp][f"{tag}_test_oracle_match_rate"] >= .9).sum())
    rng = lambda s, k, p=0: f"{q(k * s.min(), p)}--{q(k * s.max(), p)}"
    return [
        ("Abstract, C3", "up to 7.9 percentage points", "tab_e8_cascade.gain_gpt55 (max)",
         f"up to {q(100 * e8.gain_gpt55.max(), 1)} percentage points"),
        ("Abstract, VI.J, Fig. D", r"42--51\% of", "tab_cost_accuracy_tradeoff.cascade_cost_frac_of_uniform",
         rng(cost.cascade_cost_frac_of_uniform, 100) + r"\% of"),
        ("VI.J, Fig. cost_accuracy", r"36--42\%", "tab_cost_accuracy_tradeoff.esc_rate",
         rng(cost.esc_rate, 100) + r"\%"),
        ("C3", r"invocations by 58--64\%", "1 - tab_cost_accuracy_tradeoff.esc_rate",
         f"invocations by {q(100 * (1 - cost.esc_rate.max()), 0)}--{q(100 * (1 - cost.esc_rate.min()), 0)}\\%"),
        ("VI.J", r"dollar cost by 49--58\%", "tab_cost_accuracy_tradeoff.cost_saved_vs_uniform_pct",
         f"dollar cost by {rng(cost.cost_saved_vs_uniform_pct, 1)}\\%"),
        ("Abstract", r"97--100\%", "tab_cost_accuracy_tradeoff.uniform_acc", rng(cost.uniform_acc, 100) + r"\%"),
        ("VI.J", r"Uniform escalation reaches 99.6\%", "tab_cost_accuracy_tradeoff.uniform_acc (SDXL)",
         f"Uniform escalation reaches {q(100 * cost[cost.dataset == 'SDXL'].uniform_acc.iloc[0], 1)}\\%"),
        ("VI.J", r"accuracy on SDXL and 97.1\% on FLUX", "tab_cost_accuracy_tradeoff.uniform_acc (FLUX)",
         f"accuracy on SDXL and {q(100 * cost[cost.dataset == 'FLUX'].uniform_acc.iloc[0], 1)}\\% on FLUX"),
        ("VI.J", "3.0 to 7.8 percentage points", "tab_cost_accuracy_tradeoff.gap_to_uniform",
         f"{q(100 * cost.gap_to_uniform.min(), 1)} to {q(100 * cost.gap_to_uniform.max(), 1)} percentage points"),
        ("Abstract", "21 of 33 eligible configurations at $N=100$ and 25 of 33 at $N=200$",
         "tab_gate_holdout (sig, >=0.9)",
         f"{cnt('N=100', 'sig')} of {n_el} eligible configurations at $N=100$ and {cnt('N=200', 'sig')} of {n_el} at $N=200$"),
        ("VI.I.1", r"(7.3\% vs.\ 0.3\% at $N{=}50$)", "tab_gate_holdout harmful rates at N=50",
         f"({pct(gh[gh.split == 'N=50'].naive_harmful_escalation_rate.mean())} vs.\\ "
         f"{pct(gh[gh.split == 'N=50'].sig_harmful_escalation_rate.mean())} at $N{{=}}50$)"),
        ("V (confidence routing)", "more than 1.7 points less accurate", "tab_confidence_routing (max conf - dis)",
         f"more than {q(100 * cr[cr.fallback == 'semantic'].acc_diff_conf_minus_dis.max(), 1)} points less accurate"),
        ("V.B.1", "The six configurations that clear", "tab_baselines.clears_best_cheap",
         {6: "The six configurations that clear"}.get(int(base.clears_best_cheap.sum()), "COUNT CHANGED")),
    ]


FIGURES = [
    ("plot_accuracy_comparison.png", "experiments/plot_main_results_figures.py", "tab_main_results"),
    ("plot_cost_accuracy_all.png", "experiments/plot_main_results_figures.py", "tab_main_results, tab_e8_cascade"),
    ("plot_fig_d_cost_accuracy_tradeoff.png", "experiments/plot_fig_d_cost_accuracy_tradeoff.py", "tab_cost_accuracy_tradeoff"),
    ("plot_fig_f_case_mapping.png", "experiments/plot_fig_f_case_mapping.py", "tab_case_mapping"),
    ("plot_fig_a_semantic_degradation.png", "experiments/plot_fig_a_semantic_degradation.py",
     "results/e1_semantic_results.csv, pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv"),
    ("plot_fig_b_cascade_gain_scatter.png", "experiments/plot_fig_b_cascade_gain_scatter.py", "see script docstring"),
    ("plot_fig_e_resolution_shortcut.png", "experiments/plot_fig_e_resolution_shortcut.py",
     "resolution_only_control, resolution_matched_ablation"),
]

PROVENANCE = {
    "main_results": "pipeline/tables/main_results.py", "verifier_comparison": "baseline_comparison.py via pipeline/tables/baselines.py",
    "baselines": "baseline_comparison.py via pipeline/tables/baselines.py", "e8_results": "pipeline/tables/e8_cascade.py",
    "positive_cells": "pipeline/tables/e8_cascade.py", "real_image_sources": "pipeline/tables/e8_cascade.py",
    "escalation": "pipeline/tables/main_results.py", "per_generator": "pipeline/tables/per_generator.py",
    "confidence_routing": "pipeline/tables/confidence_routing.py", "cheap_path": "pipeline/tables/main_results.py",
    "cheap_trigger": "pipeline/tables/cheap_trigger.py", "limited_data": "baseline_comparison.py (CASCADE_ROWS)",
    "transfer": "pipeline/tables/transfer.py", "cross_family_deployment": "baseline_comparison.py (CASCADE_ROWS)",
    "biggan_shift": "baseline_comparison.py via pipeline/tables/baselines.py",
    "biggan_resolution": "pipeline/tables/biggan_resolution.py (reads biggan_distant_shift_*_preds.csv, *_upscaled512_preds.csv)",
    "robustness": "pipeline/tables/robustness.py (reads pipeline/robustness/e7,e9 outputs)",
    "sensitivity": "pipeline/tables/gate.py", "gate_holdout": "pipeline/tables/gate_holdout.py",
    "threeway": "pipeline/tables/gate.py", "case_mapping": "pipeline/tables/case_mapping.py",
    "cost_accuracy_tradeoff": "pipeline/tables/e8_cascade.py",
}


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _git_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip()


def build():
    tex = _norm(open(TEX).read())
    sheets, audit, long_rows = {}, [], []
    for label, fn in TABLES:
        df, rows, src = fn()
        sheets[label] = df
        for r in rows:
            audit.append({"table": label, "source_csv": src, "expected_tex_row": r, "found_in_tex": _norm(r) in tex})
        for i, rec in enumerate(df.to_dict("records")):
            key = "|".join(str(rec[c]) for c in df.columns[:3])
            for col, val in rec.items():
                long_rows.append({"table": label, "row_key": f"{i}:{key}", "column": col, "value": val,
                                  "source_csv": src.split(" ")[0] + ".csv", "source_column": col})

    inline = []
    for loc, quoted, src, recomputed in inline_checks():
        inline.append({"location": loc, "quoted_in_tex": quoted, "source": src, "recomputed": recomputed,
                       "quoted_found_in_tex": _norm(quoted) in tex, "match": quoted == recomputed})

    e8 = csv("tab_e8_cascade")
    vc = csv("tab_verifier_comparison")
    cp = csv("tab_cheap_path").set_index("dataset").n_disagree
    coverage = pd.concat([
        pd.DataFrame({"config": vc.config, "verifier": vc.verifier, "n_covered": vc.n,
                      "n_needed": vc.config.map(cp)}),
        pd.DataFrame({"config": e8.dataset + "/" + e8.semantic, "verifier": "Qwen2.5-VL-7B",
                      "n_covered": e8.n_qwen_covered, "n_needed": e8.n_disagree}),
        pd.DataFrame({"config": e8.dataset + "/" + e8.semantic, "verifier": "GPT-5.5",
                      "n_covered": e8.n_disagree, "n_needed": e8.n_disagree}),
    ], ignore_index=True)
    coverage["coverage"] = coverage.n_covered / coverage.n_needed

    n_bad = sum(not a["found_in_tex"] for a in audit) + sum(not (i["match"] and i["quoted_found_in_tex"]) for i in inline)
    readme = pd.DataFrame({"field": ["what", "generated_utc", "git_commit", "regenerate", "audit_mismatches"],
                           "value": ["Every number printed in claude_context_notes/paper_updates2.tex, by table, "
                                     "with its source CSV under routing/results/.",
                                     datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                     _git_commit(), "python routing/generate_paper_results.py "
                                     "(or python routing/build_results_bundle.py to rebuild from existing CSVs)",
                                     n_bad]})

    os.makedirs(OUT_DIR, exist_ok=True)
    with pd.ExcelWriter(os.path.join(OUT_DIR, "paper_results.xlsx")) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for label, df in sheets.items():
            df.to_excel(xw, sheet_name=label[:31], index=False)
        pd.DataFrame(inline).to_excel(xw, sheet_name="inline_numbers", index=False)
        pd.DataFrame(FIGURES, columns=["figure", "script", "inputs"]).to_excel(xw, sheet_name="figures", index=False)
        coverage.to_excel(xw, sheet_name="coverage", index=False)
        csv("tab_gate_holdout").to_excel(xw, sheet_name="gate_per_config", index=False)
        pd.DataFrame([{"table": k, "generator": v} for k, v in PROVENANCE.items()]).to_excel(
            xw, sheet_name="provenance", index=False)
        pd.DataFrame(audit).to_excel(xw, sheet_name="audit", index=False)
    pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "paper_results_long.csv"), index=False)
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("# paper_results\n\n"
                "`paper_results.xlsx` holds every number printed in the paper "
                "(`claude_context_notes/paper_updates2.tex`), one sheet per table named by its LaTeX label, "
                "plus `inline_numbers`, `figures`, `coverage`, `gate_per_config`, `provenance` and `audit`.\n\n"
                "`paper_results_long.csv` has one row per value, so `git diff` shows exactly which numbers "
                "changed between runs.\n\n"
                "The `audit` sheet renders each table's LaTeX rows from its CSV and checks that they appear "
                "verbatim in the manuscript. The `README` sheet's `audit_mismatches` must be 0 before submission.\n\n"
                "Regenerate: `python routing/generate_paper_results.py` (runs this as its last step), or "
                "`python routing/build_results_bundle.py` to rebuild from the existing CSVs. "
                "Neither calls a model or an API.\n")
    return n_bad, audit, inline


if __name__ == "__main__":
    n_bad, audit, inline = build()
    for a in audit:
        if not a["found_in_tex"]:
            print(f"[MISMATCH] {a['table']}: {a['expected_tex_row']}")
    for i in inline:
        if not (i["match"] and i["quoted_found_in_tex"]):
            print(f"[MISMATCH] inline {i['location']}: tex '{i['quoted_in_tex']}' vs recomputed '{i['recomputed']}'"
                  f" (found in tex: {i['quoted_found_in_tex']})")
    print(f"audit mismatches: {n_bad}  ({len(audit)} table rows, {len(inline)} inline numbers checked)")
    sys.exit(1 if n_bad else 0)
