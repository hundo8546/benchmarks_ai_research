import os
import warnings
from typing import Dict, List, Tuple, Optional

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL CONFIG
# ============================================================
BASE = "/workspace/benchmarks_ai_research/routing/"
EPS = 1e-12
BOOTSTRAP_N = 1000
BANDIT_LAMBDA = 0.05

# Real measured cheap-stage latency
LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_CNN_MS + LATENCY_CLIP_MS

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "font.size": 11,
})

COLORS = {
    "CNN-only": "#888780",
    "CLIP-only": "#4C78A8",
    "Disagree ensemble": "#9B59B6",
    "Entropy thresh": "#378ADD",
    "Entropy matched rate": "#5DA5DA",
    "Always escalate": "#D85A30",
    "Bandit": "#BA7517",
    "Disagree thresh": "#1D9E75",
}

METHOD_ORDER = [
    "CNN-only",
    "CLIP-only",
    "Disagree ensemble",
    "Entropy thresh",
    "Entropy matched rate",
    "Bandit",
    "Disagree thresh",
    "Always escalate",
]

FULL_BANDIT_FEATURES = [
    "confidence",
    "margin1",
    "entropy1",
    "logit",
    "y_cnn",
    "y_clip",
    "clip_proba",
    "clip_margin",
    "clip_entropy",
    "disagree",
    "prob_gap",
]

# ============================================================
# UTILS
# ============================================================
def bootstrap_ci(correct: List[int], n: int = BOOTSTRAP_N, ci: int = 95) -> Tuple[float, float]:
    scores = [np.mean(resample(correct)) for _ in range(n)]
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return lo, hi


def ensure_column(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df


def safe_mean(x) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x))


def get_qwen_filename(prefix: str) -> str:
    if prefix == "":
        return "qwen_preds.csv"
    if prefix == "sd14_":
        return "sd14_qwen_preds.csv"
    if prefix == "biggan_":
        return "biggan_qwen_preds.csv"
    raise ValueError(f"Unknown prefix: {prefix}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Core binary relations
    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)

    # CLIP uncertainty
    df["clip_margin"] = np.abs(df["clip_proba"] - 0.5)
    df["clip_entropy"] = -(
        df["clip_proba"] * np.log(df["clip_proba"] + EPS)
        + (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + EPS)
    )

    # CNN probability proxy from confidence + predicted class
    # confidence is max(p_fake, p_real)
    # reconstruct p_fake_cnn consistently
    df["p_fake_cnn"] = np.where(df["y_cnn"] == 1, df["confidence"], 1 - df["confidence"])
    df["cnn_margin"] = np.abs(df["p_fake_cnn"] - 0.5)
    df["prob_gap"] = np.abs(df["p_fake_cnn"] - df["clip_proba"])

    # Qwen probability columns if available
    df = ensure_column(df, "p_fake_qwen", np.nan)
    df = ensure_column(df, "qwen_margin", np.abs(df["p_fake_qwen"] - 0.5) if "p_fake_qwen" in df else np.nan)
    df = ensure_column(df, "qwen_input_tokens", 0)
    df = ensure_column(df, "qwen_output_tokens", 0)
    df = ensure_column(df, "y_qwen_text", "")

    # Correctness
    df["cnn_correct"] = (df["y_cnn"] == df["y_true"]).astype(int)
    df["clip_correct"] = (df["y_clip"] == df["y_true"]).astype(int)
    df["qwen_correct"] = (df["y_qwen"] == df["y_true"]).astype(int)

    # Real cost columns used everywhere
    df["latency_cnn_ms"] = LATENCY_CNN_MS
    df["latency_clip_ms"] = LATENCY_CLIP_MS
    df["latency_cheap_ms"] = LATENCY_CHEAP_MS
    df["latency_stop_ms"] = LATENCY_CHEAP_MS
    df["latency_escalate_ms"] = LATENCY_CHEAP_MS + df["latency_qwen_ms"]

    # Token cost: only incurred when Qwen is called
    df["tokens_stop_input"] = 0
    df["tokens_stop_output"] = 0
    df["tokens_esc_input"] = df["qwen_input_tokens"]
    df["tokens_esc_output"] = df["qwen_output_tokens"]

    return df


# ============================================================
# DATA LOADING
# ============================================================
def load_dataset(
    prefix: str,
    bandit_model_path: str,
    val_paths_path: str,
) -> Tuple[pd.DataFrame, object, object]:
    df_cnn = pd.read_csv(os.path.join(BASE, f"{prefix}bandit_dataset.csv"))
    df_clip = pd.read_csv(os.path.join(BASE, f"{prefix}clip_preds.csv"))
    df_qwen = pd.read_csv(os.path.join(BASE, get_qwen_filename(prefix)))

    # Keep only needed columns from auxiliary files if present
    clip_keep = [c for c in ["path", "y_clip", "clip_proba", "latency_clip_ms"] if c in df_clip.columns]
    qwen_keep = [
        c for c in [
            "path",
            "y_qwen",
            "y_qwen_text",
            "p_fake_qwen",
            "qwen_margin",
            "latency_qwen_ms",
            "qwen_input_tokens",
            "qwen_output_tokens",
        ]
        if c in df_qwen.columns
    ]

    df = df_cnn.merge(df_qwen[qwen_keep], on="path", how="inner")
    df = df.merge(df_clip[clip_keep], on="path", how="inner")

    val_paths = np.load(val_paths_path, allow_pickle=True)
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)

    # Force fixed cheap-stage latency
    if "latency_clip_ms" in df.columns:
        df["latency_clip_ms"] = LATENCY_CLIP_MS

    df = add_derived_features(df)

    bandit, scaler = joblib.load(bandit_model_path)
    return df, bandit, scaler


def load_full_dataset(prefix: str, val_paths_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df_cnn = pd.read_csv(os.path.join(BASE, f"{prefix}bandit_dataset.csv"))
    df_clip = pd.read_csv(os.path.join(BASE, f"{prefix}clip_preds.csv"))
    df_qwen = pd.read_csv(os.path.join(BASE, get_qwen_filename(prefix)))

    clip_keep = [c for c in ["path", "y_clip", "clip_proba", "latency_clip_ms"] if c in df_clip.columns]
    qwen_keep = [
        c for c in [
            "path",
            "y_qwen",
            "y_qwen_text",
            "p_fake_qwen",
            "qwen_margin",
            "latency_qwen_ms",
            "qwen_input_tokens",
            "qwen_output_tokens",
        ]
        if c in df_qwen.columns
    ]

    df = df_cnn.merge(df_qwen[qwen_keep], on="path", how="inner")
    df = df.merge(df_clip[clip_keep], on="path", how="inner")

    if "latency_clip_ms" in df.columns:
        df["latency_clip_ms"] = LATENCY_CLIP_MS

    df = add_derived_features(df)
    val_paths = np.load(val_paths_path, allow_pickle=True)
    return df, val_paths


# ============================================================
# COST + RESULT PACKAGING
# ============================================================
def make_result(
    df: pd.DataFrame,
    pred: np.ndarray,
    escalated: np.ndarray,
) -> Dict[str, List]:
    correct = (pred == df["y_true"].values).astype(int)

    mean_latency_per_sample = np.where(
        escalated == 1,
        df["latency_escalate_ms"].values,
        df["latency_stop_ms"].values,
    )
    input_tokens_per_sample = np.where(
        escalated == 1,
        df["tokens_esc_input"].values,
        df["tokens_stop_input"].values,
    )
    output_tokens_per_sample = np.where(
        escalated == 1,
        df["tokens_esc_output"].values,
        df["tokens_stop_output"].values,
    )

    return {
        "correct": correct.tolist(),
        "pred": pred.tolist(),
        "escalated": escalated.astype(int).tolist(),
        "latency_ms": mean_latency_per_sample.tolist(),
        "input_tokens": input_tokens_per_sample.tolist(),
        "output_tokens": output_tokens_per_sample.tolist(),
        "mean_latency_ms": safe_mean(mean_latency_per_sample),
        "mean_input_tokens": safe_mean(input_tokens_per_sample),
        "mean_output_tokens": safe_mean(output_tokens_per_sample),
        "escalation_rate": safe_mean(escalated),
    }


# ============================================================
# METHOD RUNNERS
# ============================================================
def run_cnn_only(df: pd.DataFrame) -> Dict:
    pred = df["y_cnn"].values
    escalated = np.zeros(len(df), dtype=int)
    return make_result(df, pred, escalated)


def run_clip_only(df: pd.DataFrame) -> Dict:
    pred = df["y_clip"].values
    escalated = np.zeros(len(df), dtype=int)
    return make_result(df, pred, escalated)


def run_always_escalate(df: pd.DataFrame) -> Dict:
    pred = df["y_qwen"].values
    escalated = np.ones(len(df), dtype=int)
    return make_result(df, pred, escalated)


def run_entropy_threshold(df: pd.DataFrame, tau: float = 0.5) -> Dict:
    escalated = (df["entropy1"].values > tau).astype(int)
    pred = np.where(escalated == 1, df["y_qwen"].values, df["y_cnn"].values)
    out = make_result(df, pred, escalated)
    out["tau"] = tau
    return out


def run_entropy_matched_rate(df: pd.DataFrame, target_rate: float) -> Dict:
    taus = np.linspace(0.0, 1.0, 1001)

    best_tau = 0.5
    best_gap = float("inf")

    for tau in taus:
        rate = float(np.mean(df["entropy1"].values > tau))
        gap = abs(rate - target_rate)
        if gap < best_gap:
            best_gap = gap
            best_tau = tau

    result = run_entropy_threshold(df, tau=best_tau)
    result["target_rate"] = target_rate
    return result


def run_disagree_thresh(df: pd.DataFrame) -> Dict:
    escalated = df["disagree"].values.astype(int)
    pred = np.where(escalated == 1, df["y_qwen"].values, df["y_cnn"].values)
    return make_result(df, pred, escalated)


def run_disagree_ensemble(df: pd.DataFrame) -> Dict:
    # Explicit CLIP-on-disagreement rule, but mechanically equal to CLIP-only
    # because y_cnn == y_clip on agreement cases.
    pred = np.where(df["disagree"].values == 1, df["y_clip"].values, df["y_clip"].values)
    escalated = np.zeros(len(df), dtype=int)
    return make_result(df, pred, escalated)


def run_bandit(df: pd.DataFrame, bandit, scaler) -> Dict:
    X = df[FULL_BANDIT_FEATURES].values
    Xs = scaler.transform(X)
    action = bandit.predict(Xs).astype(int)
    pred = np.where(action == 1, df["y_qwen"].values, df["y_cnn"].values)
    return make_result(df, pred, action)


def run_all_methods(df: pd.DataFrame, bandit, scaler) -> Dict[str, Dict]:
    results = {}

    results["CNN-only"] = run_cnn_only(df)
    results["CLIP-only"] = run_clip_only(df)
    results["Disagree ensemble"] = run_disagree_ensemble(df)
    results["Entropy thresh"] = run_entropy_threshold(df, tau=0.5)
    results["Disagree thresh"] = run_disagree_thresh(df)
    results["Bandit"] = run_bandit(df, bandit, scaler)
    results["Always escalate"] = run_always_escalate(df)

    disagree_rate = results["Disagree thresh"]["escalation_rate"]
    results["Entropy matched rate"] = run_entropy_matched_rate(df, disagree_rate)

    return results


# ============================================================
# SUMMARIES
# ============================================================
def compute_summary(results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        r = results[method]
        acc = np.mean(r["correct"])
        lo, hi = bootstrap_ci(r["correct"])
        rows.append({
            "Method": method,
            "Accuracy": acc,
            "CI_lo": lo,
            "CI_hi": hi,
            "Avg Latency (ms)": r["mean_latency_ms"],
            "Mean Input Tokens": r["mean_input_tokens"],
            "Mean Output Tokens": r["mean_output_tokens"],
            "Escalation Rate": r["escalation_rate"],
        })
    return pd.DataFrame(rows)


def compute_agree_disagree(df: pd.DataFrame) -> Dict[str, float]:
    agree_df = df[df["disagree"] == 0]
    disagree_df = df[df["disagree"] == 1]

    return {
        "n_agree": len(agree_df),
        "n_disagree": len(disagree_df),
        "agree_cnn_acc": safe_mean(agree_df["cnn_correct"]),
        "disagree_cnn_acc": safe_mean(disagree_df["cnn_correct"]),
        "agree_clip_acc": safe_mean(agree_df["clip_correct"]),
        "disagree_clip_acc": safe_mean(disagree_df["clip_correct"]),
        "agree_qwen_acc": safe_mean(agree_df["qwen_correct"]),
        "disagree_qwen_acc": safe_mean(disagree_df["qwen_correct"]),
    }


def compute_escalation_breakdown(df: pd.DataFrame, results: Dict[str, Dict]) -> pd.DataFrame:
    cases = {
        "CNN correct / CLIP agree": df[(df["cnn_correct"] == 1) & (df["disagree"] == 0)],
        "CNN correct / CLIP disagree": df[(df["cnn_correct"] == 1) & (df["disagree"] == 1)],
        "CNN wrong / CLIP agree": df[(df["cnn_correct"] == 0) & (df["disagree"] == 0)],
        "CNN wrong / CLIP disagree": df[(df["cnn_correct"] == 0) & (df["disagree"] == 1)],
    }

    rows = []
    bandit_esc = np.array(results["Bandit"]["escalated"])
    disagree_esc = np.array(results["Disagree thresh"]["escalated"])
    entropy_esc = np.array(results["Entropy thresh"]["escalated"])

    for label, subset in cases.items():
        if len(subset) == 0:
            continue
        idx = subset.index
        rows.append({
            "Case": label,
            "n": len(subset),
            "Bandit esc.": float(np.mean(bandit_esc[idx])),
            "Disagree esc.": float(np.mean(disagree_esc[idx])),
            "Entropy esc.": float(np.mean(entropy_esc[idx])),
        })

    return pd.DataFrame(rows)


def compute_per_class_table(df: pd.DataFrame, results: Dict[str, Dict], methods: List[str]) -> pd.DataFrame:
    rows = []
    fake_idx = (df["y_true"] == 1).values
    real_idx = (df["y_true"] == 0).values

    for method in methods:
        correct = np.array(results[method]["correct"])
        rows.append({
            "Method": method,
            "Fake Acc": float(np.mean(correct[fake_idx])) if fake_idx.sum() > 0 else np.nan,
            "Real Acc": float(np.mean(correct[real_idx])) if real_idx.sum() > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# LATEX HELPERS
# ============================================================
def make_latex_table(summary: pd.DataFrame, dataset_name: str, label: str) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Detection results on {dataset_name}. CI denotes 95\\% bootstrap confidence interval.}}",
        f"\\label{{tab:{label}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Accuracy & 95\% CI & Avg Latency (ms) & In Tok. & Out Tok. & Esc. Rate \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['Method']} & "
            f"{row['Accuracy']:.3f} & "
            f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] & "
            f"{row['Avg Latency (ms)']:.1f} & "
            f"{row['Mean Input Tokens']:.1f} & "
            f"{row['Mean Output Tokens']:.1f} & "
            f"{row['Escalation Rate']:.2f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def print_row(
    name: str,
    correct: List[int],
    mean_latency_ms: float,
    mean_input_tokens: float = 0.0,
    mean_output_tokens: float = 0.0,
    escalation_rate: float = 0.0,
):
    lo, hi = bootstrap_ci(correct)
    print(
        f"{name:<24} "
        f"acc={np.mean(correct):.3f} "
        f"ci=[{lo:.3f},{hi:.3f}] "
        f"lat={mean_latency_ms:.1f}ms "
        f"in_tok={mean_input_tokens:.1f} "
        f"out_tok={mean_output_tokens:.1f} "
        f"esc={escalation_rate:.3f}"
    )


# ============================================================
# PLOTTING
# ============================================================
def plot_accuracy_comparison(summary_gb, summary_sd, summary_bg):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for ax, (title, summary) in zip(axes, [
        ("GenBuster-200K-mini", summary_gb),
        ("GenImage SD 1.4", summary_sd),
        ("GenImage BigGAN", summary_bg),
    ]):
        bars = ax.bar(
            range(len(METHOD_ORDER)),
            summary["Accuracy"],
            color=[COLORS[m] for m in METHOD_ORDER],
            width=0.65,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in METHOD_ORDER], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

        for bar, row in zip(bars, summary.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{row.Accuracy:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Detection accuracy across datasets and methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_accuracy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_accuracy_comparison.png")


def plot_cost_accuracy(summary_gb, summary_sd, summary_bg):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (title, summary) in zip(axes, [
        ("GenBuster-200K-mini", summary_gb),
        ("GenImage SD 1.4", summary_sd),
        ("GenImage BigGAN", summary_bg),
    ]):
        max_cost = float(summary["Avg Latency (ms)"].max())
        for _, row in summary.iterrows():
            m = row["Method"]
            ax.scatter(row["Avg Latency (ms)"], row["Accuracy"], color=COLORS[m], s=120, zorder=5)
            ax.annotate(
                m,
                (row["Avg Latency (ms)"], row["Accuracy"]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7.5,
                color=COLORS[m],
            )

        ax.set_xlabel("Average latency (ms)")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(0, max_cost * 1.1)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Latency-accuracy frontier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_cost_accuracy_all.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_cost_accuracy_all.png")


def plot_token_pareto(summary_gb, summary_sd, summary_bg):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods_to_show = [
        "Entropy thresh",
        "Entropy matched rate",
        "Bandit",
        "Disagree thresh",
        "Always escalate",
    ]

    for ax, (title, summary) in zip(axes, [
        ("GenBuster-200K-mini", summary_gb),
        ("GenImage SD 1.4", summary_sd),
        ("GenImage BigGAN", summary_bg),
    ]):
        sub = summary[summary["Method"].isin(methods_to_show)].copy()
        sub["Total Tokens"] = sub["Mean Input Tokens"] + sub["Mean Output Tokens"]
        max_tok = max(1.0, float(sub["Total Tokens"].max()))

        for _, row in sub.iterrows():
            m = row["Method"]
            x = row["Total Tokens"]
            y = row["Accuracy"]
            ax.scatter(x, y, color=COLORS[m], s=120, zorder=5)
            ax.annotate(
                m,
                (x, y),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7.5,
                color=COLORS[m],
            )

        ax.set_xlabel("Average Qwen tokens / sample")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(0, max_tok * 1.1)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Token-space frontier for VLM-invoking methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_token_frontier_all.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_token_frontier_all.png")


def plot_agree_disagree(ad_gb, ad_sd, ad_bg):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (title, ad) in zip(axes, [
        ("GenBuster-200K-mini", ad_gb),
        ("GenImage SD 1.4", ad_sd),
        ("GenImage BigGAN", ad_bg),
    ]):
        x = np.arange(2)
        w = 0.24

        b1 = ax.bar(
            x - w,
            [ad["agree_cnn_acc"], ad["disagree_cnn_acc"]],
            w,
            label="CNN",
            color=COLORS["CNN-only"],
            alpha=0.9,
        )
        b2 = ax.bar(
            x,
            [ad["agree_clip_acc"], ad["disagree_clip_acc"]],
            w,
            label="CLIP",
            color=COLORS["CLIP-only"],
            alpha=0.9,
        )
        b3 = ax.bar(
            x + w,
            [ad["agree_qwen_acc"], ad["disagree_qwen_acc"]],
            w,
            label="Qwen",
            color=COLORS["Always escalate"],
            alpha=0.9,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"Agree\n(n={ad['n_agree']})",
            f"Disagree\n(n={ad['n_disagree']})",
        ])
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=8)

        for bar in list(b1) + list(b2) + list(b3):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Detector accuracy conditioned on CNN/CLIP agreement", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_agree_disagree_all.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_agree_disagree_all.png")


def plot_cross_dataset(summary_gb, summary_sd, summary_bg):
    fig, ax = plt.subplots(figsize=(13, 5))
    dataset_labels = ["GenBuster\n(video)", "GenImage SD 1.4\n(diffusion)", "GenImage BigGAN\n(GAN)"]
    summaries = [summary_gb, summary_sd, summary_bg]
    x = np.arange(len(dataset_labels))
    w = 0.1
    offsets = np.linspace(-3.5 * w, 3.5 * w, len(METHOD_ORDER))

    for i, method in enumerate(METHOD_ORDER):
        accs = [s[s["Method"] == method]["Accuracy"].values[0] for s in summaries]
        ax.bar(
            x + offsets[i],
            accs,
            w,
            label=method,
            color=COLORS[method],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Method accuracy across all three datasets", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_cross_dataset.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_cross_dataset.png")


def plot_escalation(esc_gb, esc_sd, esc_bg):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    for ax, (title, esc) in zip(axes, [
        ("GenBuster", esc_gb),
        ("GenImage SD 1.4", esc_sd),
        ("GenImage BigGAN", esc_bg),
    ]):
        x = np.arange(len(esc))
        w = 0.25

        ax.bar(x - w, esc["Bandit esc."], w, label="Bandit", color=COLORS["Bandit"], alpha=0.9)
        ax.bar(x, esc["Disagree esc."], w, label="Disagree thresh", color=COLORS["Disagree thresh"], alpha=0.9)
        ax.bar(x + w, esc["Entropy esc."], w, label="Entropy thresh", color=COLORS["Entropy thresh"], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(esc["Case"], fontsize=7)
        ax.set_ylim(0, 1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Escalation rate")
    fig.suptitle("Escalation behavior by case type", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_escalation_all.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_escalation_all.png")


def plot_per_class(df_gb, df_sd, df_bg, res_gb, res_sd, res_bg):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods_to_show = [
        "CNN-only",
        "Disagree ensemble",
        "Entropy matched rate",
        "Disagree thresh",
        "Always escalate",
    ]

    for ax, (title, df, res) in zip(axes, [
        ("GenBuster-200K-mini", df_gb, res_gb),
        ("GenImage SD 1.4", df_sd, res_sd),
        ("GenImage BigGAN", df_bg, res_bg),
    ]):
        fake_accs, real_accs = [], []
        fake_idx = (df["y_true"] == 1).values
        real_idx = (df["y_true"] == 0).values

        for method in methods_to_show:
            correct = np.array(res[method]["correct"])
            fake_accs.append(float(np.mean(correct[fake_idx])))
            real_accs.append(float(np.mean(correct[real_idx])))

        x = np.arange(len(methods_to_show))
        w = 0.35
        b1 = ax.bar(x - w / 2, fake_accs, w, label="Fake", color="#D85A30", alpha=0.9)
        b2 = ax.bar(x + w / 2, real_accs, w, label="Real", color="#1D9E75", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods_to_show], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

        for bar in list(b1) + list(b2):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    axes[0].set_ylabel("Per-class accuracy")
    fig.suptitle("Per-class accuracy: fake vs real detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_per_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_per_class.png")


# ============================================================
# ABLATIONS
# ============================================================
def compute_bandit_targets(df: pd.DataFrame, lam: float = BANDIT_LAMBDA) -> pd.DataFrame:
    df = df.copy()

    max_cost = float(df["latency_escalate_ms"].max())
    cost_stop = df["latency_stop_ms"] / max_cost
    cost_esc = df["latency_escalate_ms"] / max_cost

    df["r_stop"] = np.where(df["y_cnn"] == df["y_true"], 1.0, -1.0) - lam * cost_stop
    df["r_esc"] = np.where(df["y_qwen"] == df["y_true"], 1.0, -1.0) - lam * cost_esc
    df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)

    return df


def run_ablation_bandit(
    df: pd.DataFrame,
    features: List[str],
    val_paths: np.ndarray,
    lam: float = BANDIT_LAMBDA,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    df = compute_bandit_targets(df, lam=lam)

    val_mask = df["path"].isin(val_paths)
    train_mask = ~val_mask

    X = df[features].values
    y = df["target"].values

    train_idx = df[train_mask].index.values
    val_idx = df[val_mask].index.values

    if len(np.unique(y[train_idx])) < 2:
        return None, None, None, None, None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y[train_idx])

    pred = []
    escalated = []
    val_df = df.loc[val_idx].copy().reset_index(drop=True)

    for i in val_idx:
        row = df.loc[i]
        x_row = scaler.transform([df.loc[i, features].values])
        action = int(model.predict(x_row)[0])
        pred.append(row["y_qwen"] if action == 1 else row["y_cnn"])
        escalated.append(action)

    result = make_result(val_df, np.array(pred), np.array(escalated))
    return (
        float(np.mean(result["correct"])),
        result["mean_latency_ms"],
        result["escalation_rate"],
        result["mean_input_tokens"],
        result["mean_output_tokens"],
    )


def run_simple_ensemble(df: pd.DataFrame, val_paths: np.ndarray) -> Tuple[float, float, float, float]:
    val_df = df[df["path"].isin(val_paths)].reset_index(drop=True)

    cnn_prob = np.where(val_df["y_cnn"] == 1, val_df["confidence"], 1 - val_df["confidence"])
    clip_prob = val_df["clip_proba"].values
    avg_prob = (cnn_prob + clip_prob) / 2.0
    pred = (avg_prob > 0.5).astype(int)
    escalated = np.zeros(len(val_df), dtype=int)

    result = make_result(val_df, pred, escalated)
    return (
        float(np.mean(result["correct"])),
        result["mean_latency_ms"],
        result["mean_input_tokens"],
        result["mean_output_tokens"],
    )


def run_entropy_sweep(df: pd.DataFrame, taus: List[float]) -> List[Dict]:
    rows = []
    for tau in taus:
        res = run_entropy_threshold(df, tau=tau)
        rows.append({
            "Tau": tau,
            "Accuracy": float(np.mean(res["correct"])),
            "Avg Latency (ms)": res["mean_latency_ms"],
            "Avg In Tokens": res["mean_input_tokens"],
            "Avg Out Tokens": res["mean_output_tokens"],
            "Esc Rate": res["escalation_rate"],
        })
    return rows


def run_soft_disagreement_sweep(df: pd.DataFrame, thresholds: List[float]) -> List[Dict]:
    rows = []
    for thresh in thresholds:
        escalated = (df["prob_gap"].values > thresh).astype(int)
        pred = np.where(escalated == 1, df["y_qwen"].values, df["y_cnn"].values)
        res = make_result(df, pred, escalated)
        rows.append({
            "Threshold": thresh,
            "Accuracy": float(np.mean(res["correct"])),
            "Avg Latency (ms)": res["mean_latency_ms"],
            "Avg In Tokens": res["mean_input_tokens"],
            "Avg Out Tokens": res["mean_output_tokens"],
            "Esc Rate": res["escalation_rate"],
        })
    return rows


def plot_ablation_features(abl_rows: List[Dict]):
    fig, ax = plt.subplots(figsize=(11, 4))
    names = [r["Feature set"] for r in abl_rows]
    accs = [r["Accuracy"] for r in abl_rows]

    colors = ["#378ADD", "#888780", "#5DCAA5", "#BA7517", "#1D9E75", "#D85A30"]
    bars = ax.bar(
        range(len(names)),
        accs,
        color=colors[:len(names)],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        width=0.55,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title("Feature ablation: bandit routing accuracy (GenBuster)", fontsize=11, fontweight="bold")

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_ablation_features.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_ablation_features.png")


def plot_ablation_entropy(tau_rows: List[Dict]):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    taus = [r["Tau"] for r in tau_rows]
    accs = [r["Accuracy"] for r in tau_rows]
    escs = [r["Esc Rate"] for r in tau_rows]

    ax1.plot(taus, accs, marker="o", linewidth=2, color=COLORS["Entropy thresh"], label="Accuracy")
    ax2.plot(taus, escs, marker="s", linewidth=2, linestyle="--", color=COLORS["Bandit"], label="Escalation rate")

    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Entropy threshold tau")
    ax1.set_ylabel("Accuracy", color=COLORS["Entropy thresh"])
    ax2.set_ylabel("Escalation rate", color=COLORS["Bandit"])
    ax1.set_ylim(0.3, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax1.set_title("Entropy threshold sensitivity (GenBuster)", fontsize=11, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_ablation_entropy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_ablation_entropy.png")


def plot_soft_disagreement(disagree_thresh_rows: List[Dict]):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    thresh_vals = [r["Threshold"] for r in disagree_thresh_rows]
    accs_vals = [r["Accuracy"] for r in disagree_thresh_rows]
    esc_vals = [r["Esc Rate"] for r in disagree_thresh_rows]

    ax1.plot(thresh_vals, accs_vals, color=COLORS["Disagree thresh"], marker="o", linewidth=2, label="Accuracy")
    ax2.plot(thresh_vals, esc_vals, color=COLORS["Bandit"], marker="s", linewidth=2, linestyle="--", label="Escalation rate")

    ax1.axvline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.3)
    ax1.set_xlabel("Prob-gap threshold")
    ax1.set_ylabel("Accuracy", color=COLORS["Disagree thresh"])
    ax2.set_ylabel("Escalation rate", color=COLORS["Bandit"])
    ax1.set_ylim(0.3, 1.0)
    ax2.set_ylim(0.0, 1.1)
    ax1.set_title("Soft disagreement threshold sensitivity (GenBuster)", fontsize=11, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_ablation_disagree_thresh.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_ablation_disagree_thresh.png")


def plot_efficiency(efficiency_rows: List[Dict]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, dname in zip(axes, ["GenBuster", "SD14", "BigGAN"]):
        subset = [r for r in efficiency_rows if r["Dataset"] == dname]
        methods = [r["Method"] for r in subset]
        effs = [r["Acc/ms"] for r in subset]

        bars = ax.bar(
            range(len(methods)),
            effs,
            color=[COLORS[m] for m in methods],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            width=0.55,
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=8)
        ax.set_title(dname, fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy / ms")

        for bar, eff in zip(bars, effs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0001,
                f"{eff:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("Latency-normalized accuracy (Accuracy / Avg Latency)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_efficiency.png")


def train_bandit_on_dataset(df_train: pd.DataFrame, val_paths_train: np.ndarray):
    df = compute_bandit_targets(df_train, lam=BANDIT_LAMBDA)

    train_mask = ~df["path"].isin(val_paths_train)
    train_idx = df[train_mask].index.values

    X = df[FULL_BANDIT_FEATURES].values
    y = df["target"].values

    if len(np.unique(y[train_idx])) < 2:
        return None, None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y[train_idx])

    return model, scaler


def eval_transferred_bandit(model, scaler, df_test: pd.DataFrame, val_paths_test: np.ndarray):
    val_df = df_test[df_test["path"].isin(val_paths_test)].reset_index(drop=True)
    X_val = val_df[FULL_BANDIT_FEATURES].values
    Xs = scaler.transform(X_val)
    action = model.predict(Xs).astype(int)
    pred = np.where(action == 1, val_df["y_qwen"].values, val_df["y_cnn"].values)
    result = make_result(val_df, pred, action)
    return float(np.mean(result["correct"])), result["mean_latency_ms"]


def plot_transfer(transfer_rows: List[Dict]):
    train_names = ["GenBuster", "BigGAN", "SD14"]
    test_names = ["GenBuster", "BigGAN", "SD14"]
    matrix = np.zeros((3, 3))

    for r in transfer_rows:
        i = train_names.index(r["Train"])
        j = test_names.index(r["Test"])
        matrix[i, j] = r["Accuracy"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(test_names)
    ax.set_yticklabels(train_names)
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title("Cross-dataset bandit transfer accuracy", fontsize=11, fontweight="bold")

    for i in range(3):
        for j in range(3):
            label = f"{matrix[i, j]:.3f}"
            if i == j:
                label += "\n(in-domain)"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_transfer.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot_transfer.png")


# ============================================================
# MAIN
# ============================================================
print("Loading GenBuster...")
df_gb, bandit_gb, scaler_gb = load_dataset(
    "",
    os.path.join(BASE, "bandit_model.pkl"),
    os.path.join(BASE, "bandit_val_paths.npy"),
)

print("Loading SD14...")
df_sd, bandit_sd, scaler_sd = load_dataset(
    "sd14_",
    os.path.join(BASE, "bandit_model.pkl"),
    os.path.join(BASE, "sd14_bandit_val_paths.npy"),
)

print("Loading BigGAN...")
df_bg, bandit_bg, scaler_bg = load_dataset(
    "biggan_",
    os.path.join(BASE, "biggan_bandit_model.pkl"),
    os.path.join(BASE, "biggan_bandit_val_paths.npy"),
)

print("Running evaluations...")
res_gb = run_all_methods(df_gb, bandit_gb, scaler_gb)
res_sd = run_all_methods(df_sd, bandit_sd, scaler_sd)
res_bg = run_all_methods(df_bg, bandit_bg, scaler_bg)

summary_gb = compute_summary(res_gb)
summary_sd = compute_summary(res_sd)
summary_bg = compute_summary(res_bg)

ad_gb = compute_agree_disagree(df_gb)
ad_sd = compute_agree_disagree(df_sd)
ad_bg = compute_agree_disagree(df_bg)

esc_gb = compute_escalation_breakdown(df_gb, res_gb)
esc_sd = compute_escalation_breakdown(df_sd, res_sd)
esc_bg = compute_escalation_breakdown(df_bg, res_bg)

# ============================================================
# CONSOLE OUTPUT
# ============================================================
for name, s in [("GenBuster", summary_gb), ("SD14", summary_sd), ("BigGAN", summary_bg)]:
    print(f"\n=== {name} ===")
    print(s.to_string(index=False))

print("\n=== Agreement Analysis ===")
for name, ad in [("GenBuster", ad_gb), ("SD14", ad_sd), ("BigGAN", ad_bg)]:
    print(
        f"{name}: "
        f"agree n={ad['n_agree']} CNN={ad['agree_cnn_acc']:.3f} CLIP={ad['agree_clip_acc']:.3f} Qwen={ad['agree_qwen_acc']:.3f} | "
        f"disagree n={ad['n_disagree']} CNN={ad['disagree_cnn_acc']:.3f} CLIP={ad['disagree_clip_acc']:.3f} Qwen={ad['disagree_qwen_acc']:.3f}"
    )

print("\n=== Per-method rows (GenBuster) ===")
for method in METHOD_ORDER:
    r = res_gb[method]
    print_row(
        method,
        r["correct"],
        r["mean_latency_ms"],
        r["mean_input_tokens"],
        r["mean_output_tokens"],
        r["escalation_rate"],
    )

# ============================================================
# SANITY CHECKS
# ============================================================
print("\n=== Sanity checks ===")
clip_only_correct = np.array(res_gb["CLIP-only"]["correct"])
ensemble_correct = np.array(res_gb["Disagree ensemble"]["correct"])
n_differ = int(np.sum(ensemble_correct != clip_only_correct))
print(f"Qwen fake recall: {np.mean(df_gb[df_gb['y_true'] == 1]['y_qwen'] == 1):.3f}")
print(f"Qwen real recall: {np.mean(df_gb[df_gb['y_true'] == 0]['y_qwen'] == 0):.3f}")
print(f"Mean Qwen latency: {df_gb['latency_qwen_ms'].mean():.1f} ms")
print(f"Mean Qwen input tokens: {df_gb['qwen_input_tokens'].mean():.1f}")
print(f"Mean Qwen output tokens: {df_gb['qwen_output_tokens'].mean():.1f}")
print(f"Disagreement rate: {res_gb['Disagree thresh']['escalation_rate']:.3f}")
print(f"Matched-rate entropy tau: {res_gb['Entropy matched rate']['tau']:.4f}")
print(f"Ensemble vs CLIP differ on {n_differ} samples")
assert n_differ == 0, "Disagree ensemble should match CLIP-only exactly under current rule."

# ============================================================
# PLOTS
# ============================================================
plot_accuracy_comparison(summary_gb, summary_sd, summary_bg)
plot_cost_accuracy(summary_gb, summary_sd, summary_bg)
plot_token_pareto(summary_gb, summary_sd, summary_bg)
plot_agree_disagree(ad_gb, ad_sd, ad_bg)
plot_cross_dataset(summary_gb, summary_sd, summary_bg)
plot_escalation(esc_gb, esc_sd, esc_bg)
plot_per_class(df_gb, df_sd, df_bg, res_gb, res_sd, res_bg)

# ============================================================ # LATEX TABLES # ============================================================ 
print("\n\n=== LATEX TABLES ===\n") 
print(make_latex_table(summary_gb, "GenBuster-200K-mini", "genbuster")) 
print(make_latex_table(summary_sd, "GenImage SD~1.4", "sd14")) 
print(make_latex_table(summary_bg, "GenImage BigGAN", "biggan")) 
print("\n% Cross-dataset summary") 
print(r"\begin{table}[h]") 
print(r"\centering") 
print(r"\caption{Disagreement threshold accuracy across datasets and generator types.}") 
print(r"\label{tab:crossdataset}") 
print(r"\begin{tabular}{llcc}") 
print(r"\toprule") 
print(r"Dataset & Generator type & CNN-only & Disagree thresh \\")

print(r"\midrule")

for name, gtype, s in [
    ("GenBuster-200K-mini", "Video generators", summary_gb),
    ("GenImage SD~1.4",     "Diffusion images", summary_sd),
    ("GenImage BigGAN",     "GAN images",       summary_bg),
]:
    cnn = s[s["Method"] == "CNN-only"]["Accuracy"].values[0]
    dis = s[s["Method"] == "Disagree thresh"]["Accuracy"].values[0]

    print(f"{name} & {gtype} & {cnn:.3f} & {dis:.3f} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% Matched-rate entropy comparison")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Comparison between disagreement routing and entropy routing at matched escalation rate.}")
print(r"\label{tab:matched_entropy}")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"Dataset & Method & Accuracy & Avg Latency (ms) & Esc. Rate \\")
print(r"\midrule")

for dname, res in [
    ("GenBuster", res_gb),
    ("SD14",      res_sd),
    ("BigGAN",    res_bg),
]:
    for method in ["Disagree thresh", "Entropy matched rate"]:
        r = res[method]

        print(
            f"{dname} & {method} & "
            f"{np.mean(r['correct']):.3f} & "
            f"{r['mean_latency_ms']:.1f} & "
            f"{r['escalation_rate']:.2f} \\\\"
        )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% Token cost table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Average token usage for VLM-invoking methods.}")
print(r"\label{tab:tokens}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Method & Input Tokens & Output Tokens & Esc. Rate \\")
print(r"\midrule")

for method in ["Entropy thresh", "Entropy matched rate", "Bandit", "Disagree thresh", "Always escalate"]:
    r = res_gb[method]

    print(
        f"{method} & "
        f"{r['mean_input_tokens']:.1f} & "
        f"{r['mean_output_tokens']:.1f} & "
        f"{r['escalation_rate']:.2f} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n=== FINAL SANITY CHECKS ===")

# Ensemble == CLIP check
clip_only = np.array(res_gb["CLIP-only"]["correct"])
ensemble = np.array(res_gb["Disagree ensemble"]["correct"])

diff = np.sum(clip_only != ensemble)
print(f"Ensemble vs CLIP differ on {diff} samples")

# Cost consistency check
print("\nCost consistency check:")
for method in ["CNN-only", "Disagree thresh", "Always escalate"]:
    r = res_gb[method]
    print(f"{method:<20} Avg Latency = {r['mean_latency_ms']:.1f} ms")

# Escalation sanity
print("\nEscalation rates:")
for method in ["Disagree thresh", "Entropy thresh", "Entropy matched rate", "Bandit"]:
    print(f"{method:<25} {res_gb[method]['escalation_rate']:.3f}")
