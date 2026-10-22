"""
routing.pipeline.tables.case_mapping — tab_case_mapping.

Reviewer 2, point (4): "I strongly suggest summarizing these [Case 1/Case
2/Case 3/Breakdown] mappings into a unified summary table in Section VI."

This module does NOT recompute anything from raw features/predictions. It
only reads the already-written per-configuration tables under
routing/results/ (tab_main_results.csv, tab_cheap_path.csv,
tab_verifier_comparison.csv, tab_e8_results.csv,
tab_cross_family_deployment.csv, tab_biggan_shift.csv, tab_limited_data.csv,
tab_escalation.csv) and applies the mechanical case-assignment rule spelled
out in the task brief / Section III.D of claude_context_notes/paper_updates2.tex:

  Case 1 -- cheap detectors and verifier all roughly tied; disagreement rate
            p_D is tiny, so the verifier is rarely invoked and has almost no
            effect on end-to-end accuracy regardless of its own quality.
  Case 2 -- asymmetric cheap detectors: on the disagreement subset D, the
            stronger cheap detector (usually the semantic probe) already
            beats the verifier, so escalation would replace a correct cheap
            prediction with a noisier verifier one (Gain < 0).
  Case 3 -- both cheap detectors are degraded on D, AND the verifier clearly
            beats BOTH of them there (statistically, i.e. the 95% CI on
            verifier_acc - best_cheap_acc excludes 0): escalation genuinely
            helps.
  Breakdown -- f_1 and f_2 share a correlated blind spot: they AGREE while
            both are wrong, so those inputs never enter D and can never be
            routed to the verifier, no matter how good it is.

Operationalization (exactly what is "mechanical" here):
  - p_D comes from tab_cheap_path.csv's `esc_rate` column (in-domain only;
    that is the only table that reports it directly).
  - A_{3|D} ("verifier_acc") and A_{best-cheap|D} ("best_cheap_acc") +
    their paired-bootstrap 95% CI on the difference, plus the boolean
    `clears_best_cheap`, come directly from tab_verifier_comparison.csv /
    tab_e8_results.csv / tab_cross_family_deployment.csv /
    tab_biggan_shift.csv / tab_limited_data.csv (all five are slices of
    baseline_comparison.py's one underlying computation, ported in
    pipeline/tables/baselines.py). NOTE on a subtlety found while building
    this table: because disagreement is defined as y_cheap1 != y_cheap2 on a
    BINARY label, exactly one of the two cheap detectors is correct on every
    row of D, so acc(cheap1|D) + acc(cheap2|D) == 1 identically. This means
    "best_cheap_acc" (= max of the two) is NOT always the semantic probe --
    on a few cross-family rows the in-domain FFT detector is the one that
    wins on D (see the BigGAN->GenBuster flag below). The textbook Case-2
    story ("the SEMANTIC detector already beats the verifier") assumes the
    semantic probe is the stronger of the two; where that is not the literal
    winner on D, it is called out explicitly in `evidence` rather than
    silently forced into the textbook shape.
  - Case 1 is assigned when p_D <= 0.03 (BigGAN in-domain only, p_D=0.007),
    overriding the verifier-vs-best-cheap comparison, matching the paper's
    own explicit statement that BigGAN in-domain is Case 1 "because its
    artifact detector is already near ceiling" (Section VI.B) -- the point
    is that the verifier is barely ever consulted, not whether it would win
    if it were.
  - Breakdown is reported separately (it is not mutually exclusive with
    Case 1/2/3 at the level of a whole configuration -- it describes a
    subset of samples that never reach the routing decision at all) using
    tab_escalation.csv's "FFT wrong, models agree" row. THIS IS GenBuster-ONLY:
    routing/results/tab_agreement_path_audit.csv does not exist yet at the
    time this module was written (checked -- see compute_case_mapping()'s
    docstring note below), so no correlated-agreement-path audit exists for
    SD14, BigGAN, SDXL/FLUX, cross-family, biggan-shift, or limited-data.
    tab_escalation.csv (GenBuster, FFT+SigLIP2, N=1000) is used as a PARTIAL
    SUBSTITUTE per the task brief, and is flagged as such in every Breakdown
    row's evidence string.

Paper section references below use the paper's OWN in-text section labels
(confirmed against claude_context_notes/paper_updates2.tex's \\section /
\\subsection / \\subsubsection structure: III=Proposed Solution incl.
III.D=K-Stage formulation; IV=Experiments; V=Results incl. V.B=Detector
Performance, V.B.1=Baseline Comparison on the Disagreement Subset,
V.D=Ablation Studies incl. V.D.3=Probe-Degradation Sweep,
V.D.6=Cross-Family Deployment, V.D.7=Distant Generator Shift: BigGAN;
VI=Discussion incl. VI.B=The Three Cascade Cases in Practice).

CPU-only, no heavy imports: this module only ever calls pandas.read_csv on
files already written to routing/results/.
"""
import os

import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROUTING_BASE, "results")

CASE1_PD_THRESHOLD = 0.03  # disagreement rate below which escalation is "rarely invoked"


def _read(name):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _fmt(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _verifier_vs_cheap_row(dataset, configuration, verifier_acc, best_cheap_acc,
                            best_cheap_source, ci_lo, ci_hi, clears_best_cheap,
                            paper_section_reference, extra_note=""):
    """Shared Case-2-vs-Case-3 decision (used by every family except the
    p_D-driven Case-1 override and the Breakdown row)."""
    if clears_best_cheap:
        sig_note = "95% CI on (verifier - best_cheap) lies above 0"
    elif ci_hi is not None and float(ci_hi) < 0:
        sig_note = "95% CI on (verifier - best_cheap) lies below 0 (verifier significantly worse)"
    else:
        sig_note = "95% CI on (verifier - best_cheap) includes 0 (not significant)"
    evidence = (f"acc_verifier|D={_fmt(verifier_acc)} vs "
                f"acc_best_cheap|D={_fmt(best_cheap_acc)} ({best_cheap_source}), "
                f"95% CI on diff=[{_fmt(ci_lo)}, {_fmt(ci_hi)}] ({sig_note})")
    if extra_note:
        evidence += f". {extra_note}"
    case = "3" if clears_best_cheap else "2"
    return {
        "dataset": dataset,
        "configuration": configuration,
        "case": case,
        "evidence": evidence,
        "paper_section_reference": paper_section_reference,
    }


def compute_case_mapping():
    """tab_case_mapping: unified Case 1/2/3/Breakdown mapping across every
    empirical configuration in tab_main_results, tab_verifier_comparison,
    tab_e8_results, tab_cross_family_deployment, tab_biggan_shift,
    tab_limited_data, and (partially) tab_escalation.

    NOTE: routing/results/tab_agreement_path_audit.csv does NOT exist as of
    this run. The Breakdown rows below therefore use tab_escalation.csv
    (GenBuster-only, FFT+SigLIP2, N=1000) as an explicitly-flagged partial
    substitute, per the task brief. If tab_agreement_path_audit.csv is added
    later, this function should be extended to read it and this note (and
    the "PARTIAL SUBSTITUTE" wording in the Breakdown rows' evidence) should
    be revisited.
    """
    rows = []

    # ------------------------------------------------------------------
    # 1. In-domain (tab_main_results / tab_cheap_path / tab_verifier_comparison)
    #    GenBuster, SD14, BigGAN x {Qwen2.5-VL-7B, GPT-5.5}.
    # ------------------------------------------------------------------
    df_cheap_path = _read("tab_cheap_path")
    df_vc = _read("tab_verifier_comparison")
    if df_cheap_path is not None and df_vc is not None:
        pd_rate = dict(zip(df_cheap_path["dataset"], df_cheap_path["esc_rate"]))
        full_n = {"GenBuster": 259, "SD14": 532, "BigGAN": 14}  # disagreement-subset n, for partial-coverage flags
        for _, r in df_vc.iterrows():
            ds = r["config"]  # tab_verifier_comparison's "config" column IS the dataset name here
            p_d = pd_rate.get(ds)
            configuration = f"{ds} in-domain (FFT+SigLIP2 disagreement, verifier={r['verifier']})"
            partial_note = ""
            if r["n"] < full_n.get(ds, r["n"]):
                partial_note = (f"PARTIAL VERIFIER COVERAGE: n={int(r['n'])} of "
                                 f"{full_n.get(ds)} disagreement cases -- paper notes this "
                                 f"subset should not support a verifier-specific conclusion (Sec. V.B)")
            if p_d is not None and p_d <= CASE1_PD_THRESHOLD:
                evidence = (f"p_D={_fmt(p_d)} (disagreement rate on {ds}, tab_cheap_path.csv); "
                            f"cheap detectors near ceiling, verifier barely invoked. "
                            f"[For reference only, same disagreement subset: "
                            f"acc_verifier|D={_fmt(r['verifier_acc'])} vs "
                            f"acc_best_cheap|D={_fmt(r['best_cheap_acc'])} ({r['best_cheap_source']}), n={int(r['n'])}]")
                if partial_note:
                    evidence += f". {partial_note}"
                rows.append({
                    "dataset": ds,
                    "configuration": configuration,
                    "case": "1",
                    "evidence": evidence,
                    "paper_section_reference": "Sec. V.B / Table main_results; Sec. VI.B (\"Case 1 occurs on BigGAN in-domain\")",
                })
            else:
                row = _verifier_vs_cheap_row(
                    ds, configuration, r["verifier_acc"], r["best_cheap_acc"],
                    r["best_cheap_source"], r["verifier_minus_best_cheap_ci_lo"],
                    r["verifier_minus_best_cheap_ci_hi"], bool(r["clears_best_cheap"]),
                    "Sec. V.B / Table verifier_comparison; Sec. VI.B (\"describes the in-domain GenBuster and SD~1.4 results\")",
                    extra_note=partial_note)
                row["evidence"] += f" [p_D={_fmt(p_d)}]"
                rows.append(row)

    # ------------------------------------------------------------------
    # 2. SDXL/FLUX deployment shift (tab_e8_results) -- Case 3 for GPT-5.5,
    #    Case 2 for Qwen, per the paper's explicit Sec. VI.B statement.
    # ------------------------------------------------------------------
    df_e8 = _read("tab_e8_results")
    if df_e8 is not None:
        for _, r in df_e8.iterrows():
            dataset, sem = r["config"].split("/")
            configuration = f"{dataset} deployment shift, stale {sem} probe, verifier={r['verifier']}"
            row = _verifier_vs_cheap_row(
                dataset, configuration, r["verifier_acc"], r["best_cheap_acc"],
                r["best_cheap_source"], r["verifier_minus_best_cheap_ci_lo"],
                r["verifier_minus_best_cheap_ci_hi"], bool(r["clears_best_cheap"]),
                "Sec. V.B.1 / Table e8_results; Sec. VI.B (\"Case 3 occurs on the SD~1.4-to-SDXL/FLUX "
                "shift with GPT-5.5 as verifier... Replacing it with Qwen makes every gain negative\")")
            rows.append(row)

    # ------------------------------------------------------------------
    # 3. Cross-family deployment: classified mechanically, like every other cell
    #    (Case 3 iff the verifier beats the best cheap detector on the disagreement
    #    subset with a 95% CI excluding zero). BigGAN->GenBuster is the one Case 3
    #    pairing; it also beats in-domain FFT overall (tab_cross_family_deployment_cascade).
    # ------------------------------------------------------------------
    df_cfd = _read("tab_cross_family_deployment")
    if df_cfd is not None:
        for _, r in df_cfd.iterrows():
            configuration = f"cross-family {r['config']} (stale SigLIP2 + in-domain FFT, verifier={r['verifier']})"
            test_ds = r["config"].split("->")[1]
            case3 = bool(r["clears_best_cheap"]) and r["verifier_minus_best_cheap_ci_lo"] > 0
            evidence = (f"acc_verifier|D={_fmt(r['verifier_acc'])} vs "
                        f"acc_best_cheap|D={_fmt(r['best_cheap_acc'])} ({r['best_cheap_source']}), "
                        f"95% CI on diff=[{_fmt(r['verifier_minus_best_cheap_ci_lo'])}, "
                        f"{_fmt(r['verifier_minus_best_cheap_ci_hi'])}], n={int(r['n'])}.")
            rows.append({
                "dataset": test_ds,
                "configuration": configuration,
                "case": "3" if case3 else "2",
                "evidence": evidence,
                "paper_section_reference": "Sec. V.D.6 / Table cross_family_deployment; Sec. VI.B",
            })

    # ------------------------------------------------------------------
    # 4. BigGAN distant shift (tab_biggan_shift) -- Case 2 for all three
    #    verifiers per the paper's explicit text ("another Case 2 instance").
    # ------------------------------------------------------------------
    df_bs = _read("tab_biggan_shift")
    if df_bs is not None:
        for _, r in df_bs.iterrows():
            configuration = f"BigGAN distant shift (stale FFT+SigLIP2, verifier={r['verifier']})"
            row = _verifier_vs_cheap_row(
                "BigGAN", configuration, r["verifier_acc"], r["best_cheap_acc"],
                r["best_cheap_source"], r["verifier_minus_best_cheap_ci_lo"],
                r["verifier_minus_best_cheap_ci_hi"], bool(r["clears_best_cheap"]),
                "Sec. V.D.7 / Table biggan_shift; Sec. VI.B (\"the BigGAN distant shift is another "
                "Case 2 instance because the semantic probe remains stronger than every tested "
                "verifier on the routed subset\")")
            # Paper explicitly overrides to Case 2 regardless of the mechanical sign (none of the
            # three verifiers clears best-cheap here anyway, so no conflict; kept explicit for clarity).
            row["case"] = "2"
            rows.append(row)

    # ------------------------------------------------------------------
    # 5. Probe-degradation sweep (tab_limited_data) -- SD14, SigLIP2/DINOv2,
    #    6 training fractions each; Case 2 <-> Case 3 transition in practice.
    # ------------------------------------------------------------------
    df_ld = _read("tab_limited_data")
    if df_ld is not None:
        for _, r in df_ld.iterrows():
            _ds, backbone, frac = r["config"].split("/")
            configuration = f"SD14 probe-degradation sweep, {backbone}, {frac} (verifier={r['verifier']})"
            extra = ""
            if r["best_cheap_source"] not in (backbone,):
                extra = (f"NOTE: best_cheap_source={r['best_cheap_source']} (the FIXED FFT artifact "
                          f"detector), not the {backbone} semantic probe being swept here -- at this "
                          f"training fraction FFT happens to be the stronger of the two cheap "
                          f"detectors on the disagreement subset, so this is not the textbook "
                          f"'semantic probe beats verifier' Case-2 shape even though the case label "
                          f"(verifier does not clear best-cheap) is the same.")
            row = _verifier_vs_cheap_row(
                "SD14", configuration, r["verifier_acc"], r["best_cheap_acc"],
                r["best_cheap_source"], r["verifier_minus_best_cheap_ci_lo"],
                r["verifier_minus_best_cheap_ci_hi"], bool(r["clears_best_cheap"]),
                "Sec. V.D.3 / Table limited_data; Sec. VI.B (\"demonstrates the Case 2-to-Case 3 "
                "transition within a controlled experiment\")",
                extra_note=extra)
            rows.append(row)

    # ------------------------------------------------------------------
    # 6. Breakdown condition -- GenBuster-only, PARTIAL SUBSTITUTE via
    #    tab_escalation.csv (tab_agreement_path_audit.csv does not exist).
    # ------------------------------------------------------------------
    df_esc = _read("tab_escalation")
    if df_esc is not None:
        esc_map = dict(zip(df_esc["case"], df_esc["n"]))
        n_fft_wrong_agree = esc_map.get("FFT wrong, models agree")
        n_total = int(df_esc["n"].sum())
        if n_fft_wrong_agree is not None:
            frac = n_fft_wrong_agree / n_total
            rows.append({
                "dataset": "GenBuster",
                "configuration": "FFT+SigLIP2 agreement-path correlated errors (never routed to any verifier)",
                "case": "Breakdown",
                "evidence": (f"{int(n_fft_wrong_agree)}/{n_total} samples ({_fmt(frac)}) have FFT wrong "
                             f"AND SigLIP2 agreeing with it (shared blind spot, 'FFT wrong, models agree' "
                             f"row of tab_escalation.csv) -- disagree=0.00 for this case, i.e. these "
                             f"samples NEVER enter the disagreement set D and so cannot be escalated to "
                             f"any verifier regardless of verifier quality. This matches Sec. III.D's "
                             f"Breakdown condition exactly (p_D -> 0 on a correlated-error subset, "
                             f"A_cascade ~ A_agree even though the agreement set contains systematic "
                             f"errors). PARTIAL SUBSTITUTE NOTICE: routing/results/"
                             f"tab_agreement_path_audit.csv does not exist as of this run, so this row "
                             f"uses tab_escalation.csv (GenBuster, FFT+SigLIP2, N=1000) as the only "
                             f"available correlated-agreement-path evidence. No equivalent audit "
                             f"currently exists for SD14, BigGAN, SDXL/FLUX, cross-family, "
                             f"biggan-distant-shift, or the limited-data sweep -- the Breakdown "
                             f"condition's prevalence in those configurations is UNKNOWN from "
                             f"currently-generated tables."),
                "paper_section_reference": "Table escalation (Sec. V.C/\"Routing Analysis\"); "
                                            "Sec. III.D (Breakdown condition definition)",
            })

    return pd.DataFrame(rows, columns=["dataset", "configuration", "case", "evidence", "paper_section_reference"])
