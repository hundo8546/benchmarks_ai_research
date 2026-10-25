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


def _ci(lo, hi):
    b = lambda x: (r"$-$" + q(x)[1:]) if q(x).startswith("-") else q(x)
    return f"[{b(lo)}, {b(hi)}]"


def t_sizematched():
    """Table sizematched: native vs. 256 px size-matched SDXL/FLUX, GPT-5.5 cascade."""
    d = csv("tab_e8_cascade_sizematched")
    rows = []
    for ds, sem in [("SDXL", "SigLIP2"), ("SDXL", "DINOv2"), ("FLUX", "SigLIP2"), ("FLUX", "DINOv2")]:
        for variant, label in [("original", "Native"), ("sizematched256", "256~px")]:
            r = d[(d.variant == variant) & (d.dataset == ds) & (d.semantic == sem)].iloc[0]
            rows.append(row(ds, sem, label, q(r.stale_sem_acc), q(100 * r.disagree_rate, 0) + r"\%",
                            q(r.cascade_acc), f"{sgn(r.gain_over_stale)} {_ci(r.gain_ci_lo, r.gain_ci_hi)}",
                            f"{sgn(r.cascade_minus_random_routing)} "
                            f"{_ci(r.cascade_minus_random_ci_lo, r.cascade_minus_random_ci_hi)}",
                            q(r.uniform_gpt55_acc), q(100 * r.cascade_cost_share, 0) + r"\%"))
    return d, rows, "tab_e8_cascade_sizematched"


def _sm_checks():
    """inline numbers of Section sizematched / abstract / C6 / Limitations / IV.B."""
    d = csv("tab_e8_cascade_sizematched")
    s256, s512, nat = (d[d.variant == v] for v in ("sizematched256", "sizematched", "original"))
    ro = csv("resolution_only_control_sdxl_flux")
    uni = lambda x: x.drop_duplicates("dataset").set_index("dataset").uniform_gpt55_acc
    drop = 100 * (uni(nat) - uni(s256))
    words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four"}
    gp = lambda c: int((c > 0).sum())
    vm = csv("tab_e8_cascade_sizematched")  # token counts per variant
    tok = vm[vm.variant == "sizematched256"][["mean_input_tokens_real", "mean_input_tokens_fake"]].values
    cons = pd.read_csv(os.path.join(RESULTS, "batch_requests", "CONSOLIDATED_full_sdxl_flux_sizematched.csv"))
    cons256 = pd.read_csv(os.path.join(RESULTS, "batch_requests", "CONSOLIDATED_full_sdxl_flux_sizematched256.csv"))
    rng1 = lambda v: f"{q(min(v), 1)}--{q(max(v), 1)}"
    return [
        ("Abstract", "at 3.9--7.1 points, when real and fake images are size-matched", "tab_e8_cascade_sizematched (256 px gain)",
         f"at {rng1(100 * s256.gain_over_stale)} points, when real and fake images are size-matched"),
        ("C6, VI.sizematched", "a size-only classifier reaches 98.9--100.0\\%", "resolution_only_control_sdxl_flux.accuracy",
         f"a size-only classifier reaches {rng1(100 * ro.accuracy)}\\%"),
        ("V.sizematched", "with 98.9--100.0\\% accuracy", "resolution_only_control_sdxl_flux.accuracy",
         f"with {rng1(100 * ro.accuracy)}\\% accuracy"),
        ("IV.B", "image size alone separates the two classes perfectly, as it\nalso does for the COCO source",
         "resolution_only_control_sdxl_flux (voc, coco)",
         "image size alone separates the two classes perfectly, as it\nalso does for the COCO source"
         if (ro[ro.real_source.isin(["voc", "coco"])].accuracy == 1.0).all() else "NOT PERFECT"),
        ("V.sizematched", "costs the same 119 input tokens", "tab_e8_cascade_sizematched.mean_input_tokens_*",
         f"costs the same {q(tok.mean(), 0)} input tokens" if (tok == tok.flat[0]).all() else "TOKENS DIFFER"),
        ("V.sizematched", "remains significant in all four cells", "tab_e8_cascade_sizematched.gain_ci_lo (256 px)",
         f"remains significant in all {words[gp(s256.gain_ci_lo)]} cells"),
        ("V.sizematched", "beats the best cheap detector on every disagreement subset", "dis_gap_ci_lo (256 px)",
         "beats the best cheap detector on every disagreement subset" if gp(s256.dis_gap_ci_lo) == 4 else "NOT EVERY"),
        ("V.sizematched", "better in three of the four size-matched cells", "cascade_minus_random_ci_lo (256 px)",
         f"better in {words[gp(s256.cascade_minus_random_ci_lo)]} of the four size-matched cells"),
        ("V.sizematched", "as in all four cells on the native images", "cascade_minus_random_ci_lo (native)",
         f"as in all {words[gp(nat.cascade_minus_random_ci_lo)]} cells on the native images"),
        ("V.sizematched, Limitations", "falls by 2.6--3.3 points", "uniform_gpt55_acc native - 256 px",
         f"falls by {rng1(drop)} points"),
        ("V.sizematched", "costs 39--47\\% of always-GPT-5.5", "tab_e8_cascade_sizematched.cascade_cost_share (256 px)",
         f"costs {q(100 * s256.cascade_cost_share.min(), 0)}--{q(100 * s256.cascade_cost_share.max(), 0)}\\% of always-GPT-5.5"),
        ("V.sizematched", "over random routing is small, 0.4--2.3 points", "cascade_minus_random_routing (native + 256 px)",
         f"over random routing is small, {rng1(100 * pd.concat([nat, s256]).cascade_minus_random_routing)} points"),
        ("V.sizematched", "there as well (3.4--7.6 points)", "tab_e8_cascade_sizematched (512 px gain, gain_ci_lo)",
         f"there as well ({rng1(100 * s512.gain_over_stale)} points)" if gp(s512.gain_ci_lo) == 4 else "NOT SIGNIFICANT EVERYWHERE"),
        ("V.sizematched", "drives the stale FFT probe below chance on both datasets", "stale_fft_acc (512 px)",
         "drives the stale FFT probe below chance on both datasets" if (s512.stale_fft_acc < 0.5).all() else "NOT BELOW CHANCE"),
        ("V.sizematched", "no longer beats random routing in any cell", "cascade_minus_random_ci_lo (512 px)",
         "no longer beats random routing in any cell" if gp(s512.cascade_minus_random_ci_lo) == 0 else "BEATS RANDOM SOMEWHERE"),
        ("IV.B.1", "over the same 7{,}315 images", "CONSOLIDATED_full_sdxl_flux_sizematched{,256}.csv",
         f"over the same {n_tex(len(cons256))} images" if len(cons) == len(cons256) else "SIZES DIFFER"),
        ("IV.B.1", "14{,}630 calls, every one returning a valid answer", "CONSOLIDATED_full_sdxl_flux_sizematched{,256}.csv",
         f"{n_tex(len(cons) + len(cons256))} calls, every one returning a valid answer"
         if cons.y_verifier.notna().all() and cons256.y_verifier.notna().all() else "INVALID ANSWERS"),
        ("IV.B.1", "and 4\nrequests that stalled", "CONSOLIDATED_full_sdxl_flux_sizematched.csv tier",
         f"and {int((cons.tier == 'standard').sum()) + int((cons256.tier == 'standard').sum())}\nrequests that stalled"),
    ]


def t_e8_sizematched():
    """original vs. size-matched (512x512 PNG) SDXL/FLUX cascade; not printed in the paper yet."""
    return csv("tab_e8_cascade_sizematched"), [], "tab_e8_cascade_sizematched"


def t_resolution_only_sdxl_flux():
    """width/height/aspect-only classifier on SDXL/FLUX x {ImageNet, COCO, VOC} reals."""
    return csv("resolution_only_control_sdxl_flux"), [], "resolution_only_control_sdxl_flux"


def t_real_image_sources():
    d = csv("tab_real_image_sources")
    name = {"original": "PASCAL VOC (original)", "coco": "COCO (independent)"}
    rows = [row(name[r.source], r.dataset, r.semantic, q(r.agreement_path_acc), q(r.cascade_acc_gpt55),
                sgn(r.gain_over_stale_semantic, math=False)) for r in d.itertuples()]
    return d, rows, "tab_real_image_sources"


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



VER = {"Qwen2.5-VL-7B": "Qwen-7B", "Qwen2.5-VL-72B": "Qwen-72B", "GPT-5.5": "GPT-5.5"}


def _gap(point, lo, hi):
    return f"{sgn(point)}{r'$^\ast$' if lo > 0 else ''} [{sgn(lo)}, {sgn(hi)}]"


def t_verifier_subset():
    """verifier_comparison + positive_cells + biggan_shift + biggan_resolution."""
    vc, pc = csv("tab_verifier_comparison"), csv("tab_positive_cell_audit")
    br, bs = csv("tab_biggan_resolution"), csv("tab_biggan_shift")
    rows = [row(DS[r.config], VER[r.verifier], n_tex(r.n), q(r.verifier_acc), q(r.best_cheap_acc),
                _gap(r.verifier_acc - r.best_cheap_acc, r.verifier_minus_best_cheap_ci_lo,
                     r.verifier_minus_best_cheap_ci_hi)) for r in vc.itertuples()]
    rows += [row(f"{r.dataset}/{r.semantic}", "GPT-5.5", n_tex(r.n_disagree), q(r.gpt55_acc), q(r.best_cheap_acc),
                 _gap(r.gap, r.gap_ci_lo, r.gap_ci_hi)) for r in pc.itertuples()]
    for r in br.itertuples():
        rows.append(row(f"BigGAN ({'native' if r.resolution == 'native' else '512 px'})", VER[r.verifier],
                        n_tex(r.n), q(r.acc), q(r.stale_siglip2_acc), _gap(r.gap_vs_stale_siglip2, r.gap_ci_lo, r.gap_ci_hi)))
    for r in bs[bs.verifier == "Qwen2.5-VL-72B"].itertuples():
        rows.append(row("BigGAN (native)", VER[r.verifier], n_tex(r.n), q(r.verifier_acc), q(r.best_cheap_acc),
                        _gap(r.verifier_acc - r.best_cheap_acc, r.verifier_minus_best_cheap_ci_lo,
                             r.verifier_minus_best_cheap_ci_hi)))
    return vc, rows, "tab_verifier_comparison + tab_positive_cell_audit + tab_biggan_resolution + tab_biggan_shift"


def t_e8_merged():
    """e8_results + cost_accuracy_tradeoff."""
    e8 = csv("tab_e8_cascade").set_index(["dataset", "semantic"])
    cost = csv("tab_cost_accuracy_tradeoff").set_index(["dataset", "semantic"])
    rows = []
    for key in [("SDXL", "SigLIP2"), ("SDXL", "DINOv2"), ("FLUX", "SigLIP2"), ("FLUX", "DINOv2")]:
        r, c = e8.loc[key], cost.loc[key]
        rows.append(row(key[0], key[1], q(r.stale_sem_acc), f"{q(r.cascade_acc_qwen)} ({sgn(r.gain_qwen)})",
                        f"{q(r.cascade_acc_gpt55)} ({sgn(r.gain_gpt55)})", f"\\${q(c.cascade_cost_usd, 2)}",
                        f"{q(c.uniform_acc)} (\\${q(c.uniform_cost_usd, 2)})",
                        q(100 * c.cascade_cost_frac_of_uniform, 0) + r"\%"))
    return e8.reset_index(), rows, "tab_e8_cascade + tab_cost_accuracy_tradeoff"


def t_gate_merged():
    """sensitivity (same pool) + gate_holdout (held out), on one N grid."""
    s, t = csv("tab_sensitivity"), csv("tab_threeway")
    n35 = s[["table", "config", "verifier"]].drop_duplicates().shape[0]
    same = [10, 20, 50, 100, 200]
    rows = [row("Binary gate", *[f"{int((s[s.calib_N == N].binary_agreement_rate >= .9).sum())}/{n35}" for N in same], "--"),
            row("Three-way selector", *[f"{int((t[t.calib_N == N].three_way_agreement_rate >= .9).sum())}/{n35}" for N in same], "--")]
    _, held, _ = t_gate_holdout()
    for r in held:  # held-out rows have 4 cells (N=50,100,200,half); pad N=10,20
        label, rest = r.split(" & ", 1)
        rows.append(f"{label} & -- & -- & {rest}")
    return csv("tab_gate_holdout"), rows, "tab_sensitivity + tab_threeway + tab_gate_holdout"


TABLES = [  # tables printed in the paper, in paper order; sheet name = LaTeX label
    ("real_image_sources", t_real_image_sources), ("main_results", t_main_results),
    ("verifier_subset", t_verifier_subset), ("e8_results", t_e8_merged),
    ("confidence_routing", t_confidence_routing), ("cheap_trigger", t_cheap_trigger),
    ("limited_data", t_limited_data), ("transfer", t_transfer), ("cross_family_deployment", t_cross_family),
    ("case_mapping", t_case_mapping), ("gate", t_gate_merged), ("sizematched", t_sizematched),
]
SUPPLEMENTARY = [  # not printed as tables (released as sheets; key numbers quoted in the text)
    ("supp_baselines", t_baselines), ("supp_threeway", t_threeway), ("supp_escalation", t_escalation),
    ("supp_cheap_path", t_cheap_path), ("supp_per_generator", t_per_generator), ("supp_robustness", t_robustness),
    ("supp_positive_cells", t_positive_cells), ("supp_biggan_resolution", t_biggan_resolution),
    ("supp_cost_accuracy", t_cost), ("supp_e8_sizematched", t_e8_sizematched),
    ("supp_resolution_only_sdxl_flux", t_resolution_only_sdxl_flux),
]


# ── headline inline numbers: (location, text quoted in the .tex, source, recompute) ──
def _conf_gain_range():
    c, e = csv("tab_confidence_routing"), csv("tab_e8_cascade").set_index(["dataset", "semantic"])
    p = c[(c.regime != "indomain") & (c.rate_matching == "val_tau") & (c.fallback == "semantic")]
    g = [r.confidence_rule_acc - e.loc[(r.dataset, r.semantic)].stale_sem_acc for r in p.itertuples()]
    return f"by {q(100 * min(g), 1)}--{q(100 * max(g), 1)}"


def inline_checks():
    e8, cost = csv("tab_e8_cascade"), csv("tab_cost_accuracy_tradeoff")
    gh = csv("tab_gate_holdout")
    gh = gh[~gh.pool_too_small]
    cr = csv("tab_confidence_routing")
    base = csv("tab_baselines")
    n_el = gh[["table", "config", "verifier"]].drop_duplicates().shape[0]
    cnt = lambda sp, tag: int((gh[gh.split == sp][f"{tag}_test_oracle_match_rate"] >= .9).sum())
    rng = lambda s, k, p=0: f"{q(k * s.min(), p)}--{q(k * s.max(), p)}"
    cp = csv("tab_cheap_path").set_index("dataset")
    esc = csv("tab_escalation").set_index("case").n
    tw = csv("tab_threeway").drop_duplicates(["table", "config", "verifier"]).oracle_three_way_choice.value_counts()
    fft_wrong = int(esc["FFT wrong, models agree"] + esc["FFT wrong, models disagree"])
    fft_right = int(esc["FFT correct, models agree"] + esc["FFT correct, models disagree"])
    return [
        ("V.C (was Table cheap_path)", r"disagreement routes 25.9\% of GenBuster inputs, 26.6\% of SD~1.4",
         "tab_cheap_path.esc_rate",
         f"disagreement routes {q(100 * cp.esc_rate['GenBuster'], 1)}\\% of GenBuster inputs, "
         f"{q(100 * cp.esc_rate['SD14'], 1)}\\% of SD~1.4"),
        ("V.C (was Table cheap_path)", r"96.8\%, 97.8\% and 99.8\% accuracy", "tab_cheap_path.cheap_acc",
         ", ".join(f"{q(100 * cp.cheap_acc[d], 1)}\\%" for d in ["GenBuster", "SD14"]).replace(", ", "\\%, ", 0)
         + f" and {q(100 * cp.cheap_acc['BigGAN'], 1)}\\% accuracy"),
        ("V.C (was Table escalation)", "249 of FFT's 273 errors", "tab_escalation",
         f"{int(esc['FFT wrong, models disagree'])} of FFT's {fft_wrong} errors"),
        ("V.C (was Table escalation)", "10 of its 727 correct predictions", "tab_escalation",
         f"{int(esc['FFT correct, models disagree'])} of its {fft_right} correct predictions"),
        ("V.B.1 (was Table baselines)", "random baseline in 28 of the 35 configurations", "tab_baselines",
         f"random baseline in {int(base.clears_random.sum())} of the {len(base)} configurations"),
        ("V.B.1 (was Table baselines)", "majority-class baseline in 22, and the best cheap detector in only 6",
         "tab_baselines", f"majority-class baseline in {int(base.clears_majority.sum())}, and the best cheap "
         f"detector in only {int(base.clears_best_cheap.sum())}"),
        ("VI.I.2 (was Table threeway)", "it selects the semantic probe in 17 of the 35 configurations",
         "tab_threeway", f"it selects the semantic probe in {tw['cheap2']} of the {int(tw.sum())} configurations"),
        ("V.C confidence routing", "by 3.6--7.7 points across the", "tab_confidence_routing (val_tau, semantic) - stale",
         _conf_gain_range() + " points across the"),
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
    ] + _sm_checks()


FIGURES = [  # figures printed in the paper
    ("plot_accuracy_comparison.png", "experiments/plot_main_results_figures.py", "tab_main_results"),
    ("plot_fig_c_disagreement_rate.png", "experiments/plot_fig_c_disagreement_rate.py", "tab_main_results, tab_e8_cascade"),
    ("plot_cost_accuracy_all.png", "experiments/plot_main_results_figures.py", "tab_main_results, tab_e8_cascade"),
    ("plot_fig_a_semantic_degradation.png", "experiments/plot_fig_a_semantic_degradation.py",
     "tab_main_results, pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv"),
    ("plot_fig_b_cascade_gain_scatter.png", "experiments/plot_fig_b_cascade_gain_scatter.py",
     "pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv"),
]

PROVENANCE = {
    "verifier_subset": "baseline_comparison.py (verifier_comparison, biggan_shift); pipeline/tables/e8_cascade.py "
                       "(positive cells); pipeline/tables/biggan_resolution.py",
    "gate": "pipeline/tables/gate.py (same pool); pipeline/tables/gate_holdout.py (held out)",
    "case_mapping": "pipeline/tables/case_mapping.py",
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
    "sizematched": "pipeline/tables/e8_sizematched.py (native vs. 256 px rows)",
    "e8_sizematched": "pipeline/tables/e8_sizematched.py (size-matched images: pipeline/robustness/build_sizematched_sdxl_flux.py; features: pipeline/sizematched_features.py; GPT-5.5: pipeline/gpt55_sizematched.py)",
    "resolution_only_sdxl_flux": "pipeline/robustness/resolution_only_control.py (run_sdxl_flux)",
}


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _git_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip()


def build():
    tex = _norm(open(TEX).read())
    sheets, audit, long_rows = {}, [], []
    printed = {label for label, _ in TABLES}
    for label, fn in TABLES + SUPPLEMENTARY:
        df, rows, src = fn()
        sheets[label] = df
        for r in (rows if label in printed else []):
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
