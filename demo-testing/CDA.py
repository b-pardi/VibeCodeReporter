"""
full_cda.py  —  Comprehensive CDA for Empirical Software Engineering Paper
═══════════════════════════════════════════════════════════════════════════
Scans 8 domain folders, finds pre/post LLM CSV pairs, and produces:

  FIGURES (all saved to <ROOT>/CDA_results/figures/)
    F01  4×2 bubble scatter grid  (one per domain)
    F02  4×2 KDE distribution grid
    F03  Cross-domain AI-rate bar chart (grouped pre/post)
    F04  Cross-domain delta heatmap
    F05  Effect-size forest plot (Cohen's d per domain)
    F06  Confidence bucket stacked area (all domains)
    F07  Per-domain violin plot (AI probability)
    F08  CDF overlay grid (4×2)
    F09  Cliff's delta + rank-biserial bar chart
    F10  Cross-domain scatter: n_files vs Δ AI%

  STATS REPORT  (CDA_results/full_statistical_report.txt)
    • Descriptive stats (mean, median, std, IQR, skew, kurtosis, CV)
    • Normality tests: Shapiro-Wilk (n≤5000) / D'Agostino-K²
    • Homogeneity of variance: Levene's test
    • Location tests: Mann-Whitney U, Kolmogorov-Smirnov 2-sample
    • Effect sizes: Cohen's d, Cliff's Δ, rank-biserial r, CLES
    • Bootstrap 95% CI on mean difference (10,000 iterations)
    • Bonferroni-corrected p-values across 8 domains
    • Spearman correlation: repo size vs Δ AI%
    • Per-domain repo-level analysis

Usage:
    python full_cda.py [ROOT]
    # If ROOT omitted, uses <repo>/results
"""

import csv, sys, warnings, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import (
    gaussian_kde, ks_2samp, mannwhitneyu, levene,
    shapiro, normaltest, spearmanr, kruskal
)
warnings.filterwarnings("ignore")
np.random.seed(42)

# ── palette ───────────────────────────────────────────────────────────────────
PRE_C   = "#3B6FD4"
POST_C  = "#E8622A"
DOMAIN_COLORS = [
    "#3B6FD4","#E8622A","#2EAF7D","#9B59B6",
    "#E74C3C","#F39C12","#1ABC9C","#E91E8C"
]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "filename":      r["filename"],
                    "repo":          r["repo"],
                    "ai_conf":       float(r["ai_confidence"]),
                    "human_conf":    float(r["human_confidence"]),
                    "prediction":    r["prediction"].strip(),
                })
            except (KeyError, ValueError):
                continue
    return rows


def find_pairs(root: Path):
    """Return list of (domain_label, folder_name, pre_path, post_path)."""
    pairs = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        pre_files  = list(folder.glob("*prellm_predictions.csv"))
        post_files = list(folder.glob("*postllm_predictions.csv"))
        if pre_files and post_files:
            pairs.append((folder.name, folder.name, pre_files[0], post_files[0]))
    return pairs


def build_repo_table(rows):
    d = defaultdict(lambda: {"ai": 0, "human": 0, "ai_probs": []})
    for r in rows:
        d[r["repo"]]["ai_probs"].append(r["ai_conf"])
        if r["prediction"] == "AI":
            d[r["repo"]]["ai"] += 1
        else:
            d[r["repo"]]["human"] += 1
    return d

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def descriptive(a: np.ndarray) -> dict:
    n = len(a)
    if n == 0:
        return {}
    q1, q3 = np.percentile(a, [25, 75])
    mu = np.mean(a); s = np.std(a, ddof=1)
    return {
        "n": n, "mean": float(mu), "median": float(np.median(a)),
        "std": float(s), "var": float(np.var(a, ddof=1)),
        "sem": float(s / np.sqrt(n)),
        "cv":  float(s / mu) if mu else 0.0,
        "min": float(np.min(a)), "max": float(np.max(a)),
        "range": float(np.max(a) - np.min(a)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        **{f"p{p}": float(np.percentile(a, p)) for p in [1,5,10,25,50,75,90,95,99]},
        "skewness": float(stats.skew(a)),
        "kurtosis": float(stats.kurtosis(a)),  # excess kurtosis
        "high90_n":   int(np.sum(a > 0.9)),
        "high90_pct": float(np.mean(a > 0.9) * 100),
    }


def normality_test(a: np.ndarray):
    if len(a) == 0:
        return None, None, "n/a"
    if len(a) <= 5000:
        stat, p = shapiro(a[:5000])
        name = "Shapiro-Wilk"
    else:
        stat, p = normaltest(a)
        name = "D'Agostino-K²"
    return float(stat), float(p), name


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pool = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pool) if pool else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: proportion of (a>b) minus proportion of (a<b)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (len(a) * len(b)))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    try:
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
        return float(1 - (2 * u) / (len(a) * len(b)))
    except Exception:
        return 0.0


def cles(a: np.ndarray, b: np.ndarray) -> float:
    """Common Language Effect Size (P(a > b))."""
    if len(a) == 0 or len(b) == 0:
        return 0.5
    return float(np.mean(a[:, None] > b[None, :]))


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot=10000, ci=95):
    """Bootstrap CI on mean(b) - mean(a)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0
    diffs = np.array([
        np.mean(np.random.choice(b, len(b), replace=True)) -
        np.mean(np.random.choice(a, len(a), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(diffs, (100 - ci) / 2))
    hi = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    return lo, hi


def effect_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"


def cliff_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.147: return "negligible"
    if d < 0.33:  return "small"
    if d < 0.474: return "medium"
    return "large"

# ══════════════════════════════════════════════════════════════════════════════
# PER-DOMAIN STATS BUNDLE
# ══════════════════════════════════════════════════════════════════════════════

def domain_stats(domain, pre_rows, post_rows):
    pre_ap  = np.array([r["ai_conf"] for r in pre_rows])
    post_ap = np.array([r["ai_conf"] for r in post_rows])

    pre_n   = len(pre_rows);  post_n  = len(post_rows)
    pre_ai  = sum(1 for r in pre_rows  if r["prediction"] == "AI")
    post_ai = sum(1 for r in post_rows if r["prediction"] == "AI")
    pre_pct  = pre_ai  / pre_n  * 100 if pre_n  else 0
    post_pct = post_ai / post_n * 100 if post_n else 0
    d_ppt    = post_pct - pre_pct
    rel_chg  = d_ppt / pre_pct * 100 if pre_pct else float("nan")

    mw_stat, mw_p = mannwhitneyu(pre_ap, post_ap, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(pre_ap, post_ap)
    lev_stat, lev_p = levene(pre_ap, post_ap)

    cd = cohens_d(post_ap, pre_ap)
    cliff = cliffs_delta(post_ap, pre_ap)
    rb    = rank_biserial(post_ap, pre_ap)
    cl    = cles(post_ap, pre_ap)
    ci_lo, ci_hi = bootstrap_ci(pre_ap, post_ap, n_boot=5000)

    norm_pre_stat,  norm_pre_p,  norm_pre_name  = normality_test(pre_ap)
    norm_post_stat, norm_post_p, norm_post_name = normality_test(post_ap)

    pre_rt  = build_repo_table(pre_rows)
    post_rt = build_repo_table(post_rows)
    common  = set(pre_rt) & set(post_rt)
    repo_deltas = []
    repo_sizes  = []
    for repo in common:
        p = pre_rt[repo]; q = post_rt[repo]
        pt = p["ai"] + p["human"]; qt = q["ai"] + q["human"]
        if pt < 3 or qt < 3: continue
        pp = p["ai"] / pt * 100; qp = q["ai"] / qt * 100
        repo_deltas.append(qp - pp)
        repo_sizes.append(pt + qt)

    repo_deltas = np.array(repo_deltas)
    repo_sizes  = np.array(repo_sizes)
    spearman_r, spearman_p = (spearmanr(repo_sizes, repo_deltas)
                               if len(repo_deltas) > 2 else (float("nan"), float("nan")))

    return {
        "domain": domain,
        "pre_n": pre_n,   "post_n": post_n,
        "pre_ai_n": pre_ai, "post_ai_n": post_ai,
        "pre_pct": pre_pct, "post_pct": post_pct,
        "d_ppt": d_ppt,    "rel_chg": rel_chg,
        "pre_ap": pre_ap,  "post_ap": post_ap,
        "pre_desc": descriptive(pre_ap),
        "post_desc": descriptive(post_ap),
        "mw_stat": mw_stat, "mw_p": mw_p,
        "ks_stat": ks_stat, "ks_p": ks_p,
        "lev_stat": lev_stat, "lev_p": lev_p,
        "cohens_d": cd,  "cliff": cliff, "rb": rb, "cles": cl,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "norm_pre":  (norm_pre_stat,  norm_pre_p,  norm_pre_name),
        "norm_post": (norm_post_stat, norm_post_p, norm_post_name),
        "pre_rt": pre_rt, "post_rt": post_rt,
        "repo_deltas": repo_deltas, "repo_sizes": repo_sizes,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
    }

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, out_dir, name):
    fig.savefig(out_dir / name)
    plt.close(fig)
    print(f"    🖼  {name}")


# ── F01: 4×2 Bubble Scatter ───────────────────────────────────────────────────
def fig_01_bubble_grid(all_stats, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle("Per-Repository AI Detection Rate Shift: Pre vs Post LLM Era\n"
                 "(bubble size ∝ total files; above diagonal = AI increased)",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, ds in zip(axes.flat, all_stats):
        pre_rt  = ds["pre_rt"]; post_rt = ds["post_rt"]
        common  = sorted(set(pre_rt) & set(post_rt))
        pts = []
        for repo in common:
            p = pre_rt[repo]; q = post_rt[repo]
            pt = p["ai"]+p["human"]; qt = q["ai"]+q["human"]
            if pt < 3 or qt < 3: continue
            pp = p["ai"]/pt*100; qp = q["ai"]/qt*100
            pts.append((pp, qp, pt+qt))
        if not pts:
            ax.set_visible(False); continue
        px, py, sz = zip(*pts)
        px = np.array(px); py = np.array(py); sz = np.array(sz, float)
        deltas = py - px
        s_sc   = 20 + (sz / sz.max()) * 300
        lim = max(px.max(), py.max()) * 1.12
        ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, alpha=0.5, zorder=1)
        sc = ax.scatter(px, py, s=s_sc, c=deltas, cmap="RdYlGn_r",
                        vmin=-40, vmax=40, edgecolors="white",
                        linewidths=0.4, alpha=0.85, zorder=3)
        top = np.argsort(np.abs(deltas))[-4:]
        repos_list = [r for r in common if (pre_rt[r]["ai"]+pre_rt[r]["human"]) >= 3
                      and (post_rt[r]["ai"]+post_rt[r]["human"]) >= 3]
        for i in top:
            if i < len(repos_list):
                ax.annotate(repos_list[i], xy=(px[i], py[i]),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=5.5, alpha=0.8)
        ax.set_xlim(-2, lim); ax.set_ylim(-2, lim)
        ax.set_xlabel("Pre-LLM AI%", fontsize=8)
        ax.set_ylabel("Post-LLM AI%", fontsize=8)
        d_ppt = ds["d_ppt"]
        sign  = "▲" if d_ppt >= 0 else "▼"
        col   = POST_C if d_ppt >= 0 else PRE_C
        ax.set_title(f"{ds['domain']}\n{sign}{abs(d_ppt):.1f}pp  "
                     f"(pre={ds['pre_pct']:.1f}% → post={ds['post_pct']:.1f}%)",
                     fontsize=9, fontweight="bold", color=col)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02).ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, out_dir, "F01_bubble_scatter_grid.png")


# ── F02: 4×2 KDE grid ────────────────────────────────────────────────────────
def fig_02_kde_grid(all_stats, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle("AI Probability Distribution: Pre vs Post LLM Era (KDE)",
                 fontsize=13, fontweight="bold")
    x = np.linspace(0, 1, 500)
    for ax, ds in zip(axes.flat, all_stats):
        pre_ap = ds["pre_ap"]; post_ap = ds["post_ap"]
        if len(pre_ap) < 5 or len(post_ap) < 5:
            ax.set_visible(False); continue
        kp = gaussian_kde(pre_ap,  bw_method=0.08)(x)
        kq = gaussian_kde(post_ap, bw_method=0.08)(x)
        ax.fill_between(x, kp, alpha=0.2, color=PRE_C)
        ax.fill_between(x, kq, alpha=0.2, color=POST_C)
        ax.plot(x, kp, color=PRE_C,  lw=2, label=f"Pre  μ={ds['pre_desc']['mean']:.3f}")
        ax.plot(x, kq, color=POST_C, lw=2, label=f"Post μ={ds['post_desc']['mean']:.3f}")
        ax.fill_between(x, kp, kq, where=kq>kp, alpha=0.3, color=POST_C)
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_title(f"{ds['domain']}\nd={ds['cohens_d']:+.3f} ({effect_magnitude(ds['cohens_d'])})",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper center")
        ax.set_xlabel("AI Probability", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, out_dir, "F02_kde_grid.png")


# ── F03: Cross-domain grouped bar ────────────────────────────────────────────
def fig_03_crossdomain_bar(all_stats, out_dir):
    domains   = [ds["domain"] for ds in all_stats]
    pre_pcts  = [ds["pre_pct"]  for ds in all_stats]
    post_pcts = [ds["post_pct"] for ds in all_stats]
    deltas    = [ds["d_ppt"]    for ds in all_stats]

    x = np.arange(len(domains)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios":[2,1]})

    bars1 = ax1.bar(x - w/2, pre_pcts,  w, color=PRE_C,  alpha=0.88, label="Pre-LLM",  edgecolor="white")
    bars2 = ax1.bar(x + w/2, post_pcts, w, color=POST_C, alpha=0.88, label="Post-LLM", edgecolor="white")
    for bar, v in zip(bars1, pre_pcts):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}%",
                 ha="center", fontsize=8, color=PRE_C, fontweight="bold")
    for bar, v in zip(bars2, post_pcts):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}%",
                 ha="center", fontsize=8, color=POST_C, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(domains, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("AI Detection Rate (%)", fontsize=11)
    ax1.set_title("Cross-Domain AI Detection Rate: Pre vs Post LLM Era", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    colors = [POST_C if d >= 0 else PRE_C for d in deltas]
    ax2.bar(x, deltas, color=colors, alpha=0.88, edgecolor="white")
    ax2.axhline(0, color="black", lw=1)
    for i, d in enumerate(deltas):
        ax2.text(i, d + (0.2 if d >= 0 else -0.5), f"{d:+.2f}pp",
                 ha="center", fontsize=8, fontweight="bold",
                 color=POST_C if d >= 0 else PRE_C)
    ax2.set_xticks(x); ax2.set_xticklabels(domains, rotation=20, ha="right", fontsize=10)
    ax2.set_ylabel("Δ AI% (pp)", fontsize=11)
    ax2.set_title("Delta: Post − Pre", fontsize=11)

    fig.tight_layout()
    save(fig, out_dir, "F03_crossdomain_bar.png")


# ── F04: Heatmap ──────────────────────────────────────────────────────────────
def fig_04_heatmap(all_stats, out_dir):
    domains = [ds["domain"] for ds in all_stats]
    metrics = {
        "Pre AI%":        [ds["pre_pct"]                     for ds in all_stats],
        "Post AI%":       [ds["post_pct"]                    for ds in all_stats],
        "Δ pp":           [ds["d_ppt"]                       for ds in all_stats],
        "Rel. Δ%":        [ds["rel_chg"]                     for ds in all_stats],
        "Cohen's d":      [ds["cohens_d"]                    for ds in all_stats],
        "Cliff's Δ":      [ds["cliff"]                       for ds in all_stats],
        "KS stat":        [ds["ks_stat"]                     for ds in all_stats],
        "Pre mean conf":  [ds["pre_desc"]["mean"]            for ds in all_stats],
        "Post mean conf": [ds["post_desc"]["mean"]           for ds in all_stats],
        "Δ mean conf":    [ds["post_desc"]["mean"]-ds["pre_desc"]["mean"] for ds in all_stats],
    }
    mat = np.array(list(metrics.values()))

    # normalize each row to [-1,1] for display
    mat_norm = np.zeros_like(mat)
    for i, row in enumerate(mat):
        rng = np.max(np.abs(row))
        mat_norm[i] = row / rng if rng else row

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics.keys(), fontsize=10)

    for i in range(len(metrics)):
        for j in range(len(domains)):
            raw_val = list(metrics.values())[i][j]
            ax.text(j, i, f"{raw_val:.2f}", ha="center", va="center", fontsize=8,
                    color="black")

    plt.colorbar(im, ax=ax, label="Row-normalised value", shrink=0.8)
    ax.set_title("Cross-Domain Metric Heatmap (values annotated, rows row-normalised for colour)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "F04_metric_heatmap.png")


# ── F05: Forest plot (Cohen's d) ──────────────────────────────────────────────
def fig_05_forest(all_stats, out_dir):
    domains = [ds["domain"] for ds in all_stats]
    ds_vals = [ds["cohens_d"] for ds in all_stats]
    ci_los  = [ds["ci_lo"]    for ds in all_stats]
    ci_his  = [ds["ci_hi"]    for ds in all_stats]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(domains))
    colors = [POST_C if d > 0 else PRE_C for d in ds_vals]
    ax.barh(y, ds_vals, height=0.5, color=colors, alpha=0.8, zorder=3)

    for i, (lo, hi, d) in enumerate(zip(ci_los, ci_his, ds_vals)):
        ax.plot([lo, hi], [i, i], color="black", lw=2, zorder=4)
        ax.plot([lo, lo], [i-0.15, i+0.15], color="black", lw=2, zorder=4)
        ax.plot([hi, hi], [i-0.15, i+0.15], color="black", lw=2, zorder=4)
        mag = effect_magnitude(d)
        ax.text(max(hi, d) + 0.01, i, f" d={d:+.3f} ({mag})",
                va="center", fontsize=9)

    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.axvspan(-0.2, 0.2, alpha=0.05, color="gray", label="Negligible (|d|<0.2)")
    ax.set_yticks(y); ax.set_yticklabels(domains, fontsize=11)
    ax.set_xlabel("Cohen's d  (Post − Pre)", fontsize=11)
    ax.set_title("Forest Plot — Effect Size per Domain\n(error bars = 95% bootstrap CI on mean difference)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "F05_forest_plot.png")


# ── F06: Confidence bucket stacked area ──────────────────────────────────────
def fig_06_bucket_area(all_stats, out_dir):
    bucket_edges  = [0, .5, .6, .7, .8, .9, 1.01]
    bucket_labels = ["0.0–0.5\n(Human)", "0.5–0.6", "0.6–0.7",
                     "0.7–0.8", "0.8–0.9", "0.9–1.0\n(AI)"]
    domains = [ds["domain"] for ds in all_stats]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle("Confidence Bucket Distribution Across Domains", fontsize=13, fontweight="bold")

    bucket_colors = ["#2ECC71","#F1C40F","#E67E22","#E74C3C","#9B59B6","#3498DB"]

    for ax, era, era_key in [(axes[0], "Pre-LLM", "pre_ap"), (axes[1], "Post-LLM", "post_ap")]:
        data = []
        for ds in all_stats:
            ap = ds[era_key]
            total = len(ap)
            row = []
            for i in range(len(bucket_edges)-1):
                lo, hi = bucket_edges[i], bucket_edges[i+1]
                row.append(np.sum((ap >= lo) & (ap < hi)) / total * 100 if total else 0)
            data.append(row)
        data = np.array(data).T  # shape (6, n_domains)

        bottom = np.zeros(len(domains))
        x = np.arange(len(domains))
        for i, (row, lbl, col) in enumerate(zip(data, bucket_labels, bucket_colors)):
            ax.bar(x, row, bottom=bottom, color=col, label=lbl, alpha=0.88, edgecolor="white")
            for j, (v, b) in enumerate(zip(row, bottom)):
                if v > 4:
                    ax.text(j, b + v/2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += row

        ax.set_xticks(x); ax.set_xticklabels(domains, rotation=25, ha="right", fontsize=9)
        ax.set_title(era, fontsize=12, fontweight="bold",
                     color=PRE_C if era=="Pre-LLM" else POST_C)
        ax.set_ylabel("% of files", fontsize=10)

    axes[1].legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.18, 1))
    fig.tight_layout()
    save(fig, out_dir, "F06_bucket_stacked_bar.png")


# ── F07: Violin grid ─────────────────────────────────────────────────────────
def fig_07_violin(all_stats, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle("AI Probability Distribution — Violin + Box: Pre vs Post LLM Era",
                 fontsize=13, fontweight="bold")

    for ax, ds in zip(axes.flat, all_stats):
        pre_ap = ds["pre_ap"]; post_ap = ds["post_ap"]
        data = [pre_ap, post_ap]
        positions = [1, 2]
        vp = ax.violinplot(data, positions=positions, widths=0.65,
                           showmedians=False, showextrema=False)
        for body, col in zip(vp["bodies"], [PRE_C, POST_C]):
            body.set_facecolor(col); body.set_alpha(0.45)
        ax.boxplot(data, positions=positions, widths=0.18, patch_artist=True, notch=True,
                   medianprops=dict(color="white", lw=2),
                   boxprops=dict(facecolor="none"),
                   whiskerprops=dict(color="black"),
                   capprops=dict(color="black"),
                   flierprops=dict(marker=".", markersize=1.5, alpha=0.25))
        ax.set_xticks(positions); ax.set_xticklabels(["Pre", "Post"], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{ds['domain']}\n"
                     f"pre={ds['pre_pct']:.1f}% → post={ds['post_pct']:.1f}%",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("AI Probability", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, out_dir, "F07_violin_grid.png")


# ── F08: CDF grid ─────────────────────────────────────────────────────────────
def fig_08_cdf_grid(all_stats, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle("Empirical CDF of AI Probability — Pre vs Post LLM Era",
                 fontsize=13, fontweight="bold")
    for ax, ds in zip(axes.flat, all_stats):
        for arr, col, lbl in [(ds["pre_ap"], PRE_C, "Pre"), (ds["post_ap"], POST_C, "Post")]:
            s = np.sort(arr)
            ax.plot(s, np.arange(1, len(s)+1)/len(s), color=col, lw=2, label=lbl)
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(f"{ds['domain']}\nKS={ds['ks_stat']:.4f} p={ds['ks_p']:.2e}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("AI Prob", fontsize=8); ax.set_ylabel("CDF", fontsize=8)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, out_dir, "F08_cdf_grid.png")


# ── F09: Effect size bar chart ────────────────────────────────────────────────
def fig_09_effect_sizes(all_stats, out_dir):
    domains  = [ds["domain"]  for ds in all_stats]
    cd_vals  = [ds["cohens_d"] for ds in all_stats]
    cl_vals  = [ds["cliff"]    for ds in all_stats]
    rb_vals  = [ds["rb"]       for ds in all_stats]
    cles_vals= [ds["cles"]     for ds in all_stats]

    x = np.arange(len(domains)); w = 0.2
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Cohen's d + Cliff's delta
    ax = axes[0]
    ax.bar(x - w, cd_vals,  w*1.8, label="Cohen's d",    color="#3B6FD4", alpha=0.85, edgecolor="white")
    ax.bar(x + w, cl_vals,  w*1.8, label="Cliff's Δ",    color="#E8622A", alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    for thresh, ls, lbl in [(0.2,"--","small"),(0.5,":","medium"),(0.8,"-.","large")]:
        ax.axhline(thresh, color="gray", lw=0.8, ls=ls, alpha=0.5)
        ax.axhline(-thresh, color="gray", lw=0.8, ls=ls, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(domains, rotation=20, ha="right", fontsize=10)
    ax.set_title("Effect Sizes per Domain", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.set_ylabel("Effect size")

    # Bottom: CLES
    ax2 = axes[1]
    colors = [POST_C if v > 0.5 else PRE_C for v in cles_vals]
    ax2.bar(x, cles_vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(0.5, color="black", lw=1.2, ls="--", label="CLES=0.5 (no effect)")
    for i, v in enumerate(cles_vals):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(domains, rotation=20, ha="right", fontsize=10)
    ax2.set_title("Common Language Effect Size  P(Post > Pre)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("CLES"); ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    save(fig, out_dir, "F09_effect_sizes.png")


# ── F10: n_files vs Δ AI% scatter ─────────────────────────────────────────────
def fig_10_size_vs_delta(all_stats, out_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, ds in enumerate(all_stats):
        rd = ds["repo_deltas"]; rs = ds["repo_sizes"]
        if len(rd) == 0: continue
        ax.scatter(rs, rd, s=40, alpha=0.55, color=DOMAIN_COLORS[i],
                   label=ds["domain"], edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Repository size (pre + post files)", fontsize=11)
    ax.set_ylabel("Δ AI Detection Rate (pp)", fontsize=11)
    ax.set_title("Repository Size vs AI Rate Change\n"
                 "(all domains overlaid — Spearman ρ annotated per domain)",
                 fontsize=12, fontweight="bold")

    # Spearman annotations
    txt = []
    for ds in all_stats:
        r = ds["spearman_r"]; p = ds["spearman_p"]
        if not math.isnan(r):
            txt.append(f"{ds['domain'][:8]}: ρ={r:.3f} p={p:.2e}")
    ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes,
            fontsize=7.5, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, "F10_size_vs_delta.png")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_stats, out_dir):
    path = out_dir / "full_statistical_report.txt"
    n_domains = len(all_stats)

    # Bonferroni correction
    mw_ps  = [ds["mw_p"]  for ds in all_stats]
    ks_ps  = [ds["ks_p"]  for ds in all_stats]
    bonf_mw = [min(p * n_domains, 1.0) for p in mw_ps]
    bonf_ks = [min(p * n_domains, 1.0) for p in ks_ps]

    SEP  = "─" * 76
    DSEP = "═" * 76

    with open(path, "w", encoding="utf-8") as f:
        def w(s=""): f.write(s + "\n")
        def sec(t):  w(f"\n{DSEP}\n  {t}\n{DSEP}")
        def sub(t):  w(f"\n  {SEP}\n  {t}\n  {SEP}")

        w(DSEP)
        w("  COMPREHENSIVE CDA REPORT — AI Code Detection: Pre vs Post LLM Era")
        w(f"  Domains   : {n_domains}")
        w(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(DSEP)

        # ── 1. Overview table ──────────────────────────────────────────────
        sec("1. DATASET OVERVIEW")
        w(f"\n  {'DOMAIN':<22}  {'PRE_N':>8}  {'POST_N':>8}  {'PRE_AI%':>8}  {'POST_AI%':>9}  {'Δ pp':>8}  {'Rel%':>8}")
        w(f"  {SEP}")
        for ds in all_stats:
            w(f"  {ds['domain']:<22}  {ds['pre_n']:>8,}  {ds['post_n']:>8,}  "
              f"{ds['pre_pct']:>7.2f}%  {ds['post_pct']:>8.2f}%  "
              f"{ds['d_ppt']:>+7.2f}pp  {ds['rel_chg']:>+7.1f}%")

        # ── 2. Descriptive stats per domain ───────────────────────────────
        sec("2. DESCRIPTIVE STATISTICS — AI PROBABILITY DISTRIBUTION")
        for ds in all_stats:
            sub(f"{ds['domain']}")
            w(f"\n  {'METRIC':<20}  {'PRE-LLM':>12}  {'POST-LLM':>12}  {'DELTA':>12}")
            w(f"  {'─'*60}")
            pre_d = ds["pre_desc"]; post_d = ds["post_desc"]
            for m in ["n","mean","median","std","var","sem","cv","min","max",
                      "range","q1","q3","iqr","p1","p5","p10","p25","p50",
                      "p75","p90","p95","p99","skewness","kurtosis",
                      "high90_n","high90_pct"]:
                pv = pre_d.get(m, 0); qv = post_d.get(m, 0)
                if m == "n":
                    w(f"  {m:<20}  {int(pv):>12,}  {int(qv):>12,}  {int(qv-pv):>+12,}")
                else:
                    w(f"  {m:<20}  {pv:>12.6f}  {qv:>12.6f}  {qv-pv:>+12.6f}")

        # ── 3. Normality tests ────────────────────────────────────────────
        sec("3. NORMALITY TESTS")
        w(f"\n  {'DOMAIN':<22}  {'ERA':<6}  {'TEST':<18}  {'STAT':>10}  {'p-value':>12}  {'NORMAL?':>8}")
        w(f"  {SEP}")
        for ds in all_stats:
            for era, (stat, p, name) in [("Pre",  ds["norm_pre"]),
                                          ("Post", ds["norm_post"])]:
                normal = "YES" if (p or 0) > 0.05 else "NO"
                w(f"  {ds['domain']:<22}  {era:<6}  {name:<18}  "
                  f"{(stat or 0):>10.4f}  {(p or 0):>12.2e}  {normal:>8}")

        # ── 4. Homogeneity of variance ────────────────────────────────────
        sec("4. HOMOGENEITY OF VARIANCE — Levene's Test")
        w(f"\n  {'DOMAIN':<22}  {'STAT':>10}  {'p-value':>12}  {'EQUAL VAR?':>12}")
        w(f"  {SEP}")
        for ds in all_stats:
            eq = "YES" if ds["lev_p"] > 0.05 else "NO"
            w(f"  {ds['domain']:<22}  {ds['lev_stat']:>10.4f}  {ds['lev_p']:>12.2e}  {eq:>12}")

        # ── 5. Location tests ─────────────────────────────────────────────
        sec("5. LOCATION TESTS  (Pre vs Post AI Probability)")
        w(f"\n  {'DOMAIN':<22}  {'MWU stat':>12}  {'MWU p':>12}  {'MWU Bonf':>12}  "
          f"{'KS stat':>10}  {'KS p':>12}  {'KS Bonf':>12}  {'SIG?':>6}")
        w(f"  {SEP}")
        for ds, bm, bk in zip(all_stats, bonf_mw, bonf_ks):
            sig = "✓" if bm < 0.05 else "✗"
            w(f"  {ds['domain']:<22}  {ds['mw_stat']:>12.1f}  {ds['mw_p']:>12.2e}  "
              f"{bm:>12.2e}  {ds['ks_stat']:>10.4f}  {ds['ks_p']:>12.2e}  "
              f"{bk:>12.2e}  {sig:>6}")
        w(f"\n  Bonferroni correction: α/k = 0.05/{n_domains} = {0.05/n_domains:.4f}")

        # ── 6. Effect sizes ───────────────────────────────────────────────
        sec("6. EFFECT SIZES")
        w(f"\n  {'DOMAIN':<22}  {'Cohen d':>9}  {'Mag':>10}  {'Cliff Δ':>9}  {'Mag':>10}  "
          f"{'Rank-bis r':>12}  {'CLES':>8}  {'95% CI (Δmean)':>20}")
        w(f"  {SEP}")
        for ds in all_stats:
            w(f"  {ds['domain']:<22}  {ds['cohens_d']:>+9.4f}  "
              f"{effect_magnitude(ds['cohens_d']):>10}  "
              f"{ds['cliff']:>+9.4f}  "
              f"{cliff_magnitude(ds['cliff']):>10}  "
              f"{ds['rb']:>+12.4f}  "
              f"{ds['cles']:>8.4f}  "
              f"[{ds['ci_lo']:+.4f}, {ds['ci_hi']:+.4f}]")

        # ── 7. Kruskal-Wallis across domains ──────────────────────────────
        sec("7. KRUSKAL-WALLIS TEST — Are all post-LLM distributions equal?")
        post_arrays = [ds["post_ap"] for ds in all_stats if len(ds["post_ap"]) > 0]
        if len(post_arrays) >= 2:
            kw_stat, kw_p = kruskal(*post_arrays)
            w(f"\n  H statistic : {kw_stat:.4f}")
            w(f"  p-value     : {kw_p:.2e}")
            w(f"  Verdict     : {'Distributions DIFFER significantly across domains' if kw_p < 0.05 else 'No significant difference across domains'}")

        # ── 8. Pairwise domain comparisons (post-LLM) ─────────────────────
        sec("8. PAIRWISE DOMAIN COMPARISONS — Post-LLM AI Probability")
        domain_names = [ds["domain"] for ds in all_stats]
        pairs = list(combinations(range(len(all_stats)), 2))
        w(f"\n  {'DOMAIN A':<22}  {'DOMAIN B':<22}  {'MWU p':>12}  {'KS p':>12}  {'Cohen d':>9}  {'SIG?':>6}")
        w(f"  {SEP}")
        for i, j in pairs:
            a = all_stats[i]["post_ap"]; b = all_stats[j]["post_ap"]
            if len(a) == 0 or len(b) == 0: continue
            _, p_mw = mannwhitneyu(a, b, alternative="two-sided")
            _, p_ks = ks_2samp(a, b)
            cd = cohens_d(a, b)
            bonf_p = min(p_mw * len(pairs), 1.0)
            sig = "✓" if bonf_p < 0.05 else "✗"
            w(f"  {domain_names[i]:<22}  {domain_names[j]:<22}  "
              f"{p_mw:>12.2e}  {p_ks:>12.2e}  {cd:>+9.4f}  {sig:>6}")
        w(f"\n  Bonferroni α: {0.05/len(pairs):.5f}")

        # ── 9. Repo-level Spearman ────────────────────────────────────────
        sec("9. SPEARMAN CORRELATION — Repository Size vs Δ AI%")
        w(f"\n  {'DOMAIN':<22}  {'ρ':>8}  {'p-value':>12}  {'Repos':>7}  {'Interpretation':>20}")
        w(f"  {SEP}")
        for ds in all_stats:
            r = ds["spearman_r"]; p = ds["spearman_p"]
            n = len(ds["repo_deltas"])
            if math.isnan(r):
                interp = "insufficient data"
            elif abs(r) < 0.1:  interp = "negligible"
            elif abs(r) < 0.3:  interp = "weak"
            elif abs(r) < 0.5:  interp = "moderate"
            else:               interp = "strong"
            p_str = f"{p:.2e}" if not math.isnan(p) else "n/a"
            r_str = f"{r:+.4f}" if not math.isnan(r) else "n/a"
            w(f"  {ds['domain']:<22}  {r_str:>8}  {p_str:>12}  {n:>7,}  {interp:>20}")

        # ── 10. Per-domain repo breakdown ─────────────────────────────────
        sec("10. PER-DOMAIN PER-REPOSITORY BREAKDOWN  (sorted by |Δ AI%|)")
        for ds in all_stats:
            sub(f"{ds['domain']}")
            pre_rt = ds["pre_rt"]; post_rt = ds["post_rt"]
            common = set(pre_rt) & set(post_rt)
            rows_out = []
            for repo in common:
                p = pre_rt[repo]; q = post_rt[repo]
                pt = p["ai"]+p["human"]; qt = q["ai"]+q["human"]
                if pt == 0 or qt == 0: continue
                pp = p["ai"]/pt*100; qp = q["ai"]/qt*100
                pc = np.mean(p["ai_probs"]); qc = np.mean(q["ai_probs"])
                rows_out.append((repo, pt, qt, pp, qp, qp-pp, pc, qc))
            w(f"\n  {'REPO':<35}  {'PRE_T':>6}  {'POST_T':>6}  "
              f"{'PRE%':>7}  {'POST%':>7}  {'Δ pp':>7}  {'PRE_CONF':>9}  {'POST_CONF':>9}")
            w(f"  {'─'*100}")
            for repo, pt, qt, pp, qp, dp, pc, qc in sorted(rows_out, key=lambda x: -abs(x[5])):
                w(f"  {repo:<35}  {pt:>6,}  {qt:>6,}  "
                  f"{pp:>6.1f}%  {qp:>6.1f}%  {dp:>+6.1f}pp  {pc:>9.4f}  {qc:>9.4f}")

        w(f"\n{DSEP}")
        w(f"  END OF REPORT — {n_domains} domains")
        w(f"  Generated: {datetime.now().isoformat()}")
        w(DSEP)

    print(f"    📄  full_statistical_report.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _repo_root = Path(__file__).resolve().parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else (_repo_root / "results")
    pairs = find_pairs(root)
    if not pairs:
        print("No paired CSVs found. Check folder structure."); return

    out_dir = root / "CDA_results"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(pairs)} domain(s): {[p[0] for p in pairs]}")
    print(f"Output → {out_dir}\n")

    print("Loading & computing stats...")
    all_stats = []
    for domain, _, pre_path, post_path in pairs:
        pre_rows  = load_csv(pre_path)
        post_rows = load_csv(post_path)
        if not pre_rows or not post_rows:
            print(f"  ⚠️  Skipping {domain} — empty CSV"); continue
        print(f"  {domain:<25}  pre={len(pre_rows):,}  post={len(post_rows):,}")
        all_stats.append(domain_stats(domain, pre_rows, post_rows))

    if not all_stats:
        print("No data to analyse."); return

    print("\nGenerating figures...")
    fig_01_bubble_grid(all_stats, fig_dir)
    fig_02_kde_grid(all_stats, fig_dir)
    fig_03_crossdomain_bar(all_stats, fig_dir)
    fig_04_heatmap(all_stats, fig_dir)
    fig_05_forest(all_stats, fig_dir)
    fig_06_bucket_area(all_stats, fig_dir)
    fig_07_violin(all_stats, fig_dir)
    fig_08_cdf_grid(all_stats, fig_dir)
    fig_09_effect_sizes(all_stats, fig_dir)
    fig_10_size_vs_delta(all_stats, fig_dir)

    print("\nWriting statistical report...")
    write_report(all_stats, out_dir)

    print(f"\n✅  All done → {out_dir}")


if __name__ == "__main__":
    main()