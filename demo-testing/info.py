import csv, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu
import warnings; warnings.filterwarnings("ignore")

# ── palette ───────────────────────────────────────────────────────────────────
PRE_C  = "#3B6FD4"
POST_C = "#E8622A"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.18,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "filename":         r["filename"],
                    "repo":             r["repo"],
                    "ai_confidence":    float(r["ai_confidence"]),
                    "human_confidence": float(r["human_confidence"]),
                    "prediction":       r["prediction"].strip(),
                })
            except (KeyError, ValueError):
                continue
    return rows


def find_pairs(search_dir: Path):
    pre  = {f.name.replace("-prellm_predictions.csv", ""): f
            for f in search_dir.rglob("*-prellm_predictions.csv")}
    post = {f.name.replace("-postllm_predictions.csv", ""): f
            for f in search_dir.rglob("*-postllm_predictions.csv")}
    return [(d, pre[d], post[d]) for d in sorted(set(pre) & set(post))]


def build_repo_table(rows):
    d = defaultdict(lambda: {"ai": 0, "human": 0, "ai_probs": [], "human_probs": []})
    for r in rows:
        k = r["repo"]
        d[k]["ai_probs"].append(r["ai_confidence"])
        d[k]["human_probs"].append(r["human_confidence"])
        if r["prediction"] == "AI":
            d[k]["ai"] += 1
        else:
            d[k]["human"] += 1
    return d


def bucket_counts(probs, total):
    edges  = [0, .5, .6, .7, .8, .9, 1.01]
    labels = ["0.0–0.5 (Human)", "0.5–0.6", "0.6–0.7",
              "0.7–0.8", "0.8–0.9", "0.9–1.0 (High conf AI)"]
    out = []
    for i, lbl in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        n = int(np.sum((probs >= lo) & (probs < hi)))
        out.append((lbl, n, n / total * 100 if total else 0))
    return out


# ── report text helpers ───────────────────────────────────────────────────────

def stats_block(rows, label):
    """Reproduce the exact output format from the original pipeline."""
    ai_rows  = [r for r in rows if r["prediction"] == "AI"]
    hum_rows = [r for r in rows if r["prediction"] == "Human"]
    all_ai   = np.array([r["ai_confidence"]    for r in rows])
    ai_conf  = np.array([r["ai_confidence"]    for r in ai_rows])
    hum_conf = np.array([r["human_confidence"] for r in hum_rows])
    total = len(rows);  n_ai = len(ai_rows);  n_hum = len(hum_rows)

    SEP = "  " + "─" * 70
    L = []
    L.append(f"\n{'='*70}")
    L.append(f"  RESULTS: {label}")
    L.append(f"{'='*70}")
    L.append(f"")
    L.append(f"  Total files      : {total:,}")
    L.append(f"  AI-generated     : {n_ai:,}  ({n_ai/total*100:.2f}%)")
    L.append(f"  Human-written    : {n_hum:,}  ({n_hum/total*100:.2f}%)")
    L.append(f"")
    L.append(SEP)
    L.append(f"  CONFIDENCE SCORE DISTRIBUTIONS")
    L.append(SEP)

    for arr, lbl in [
        (ai_conf,  "AI predictions   (AI confidence)"),
        (hum_conf, "Human predictions (Human confidence)"),
        (all_ai,   "All files         (AI probability)"),
    ]:
        if len(arr) == 0:
            continue
        hc = int(np.sum(arr > 0.9))
        L.append(f"")
        L.append(f"  [{lbl}]")
        L.append(f"    n         : {len(arr):,}")
        L.append(f"    mean      : {np.mean(arr):.4f}")
        L.append(f"    median    : {np.median(arr):.4f}")
        L.append(f"    std       : {np.std(arr):.4f}")
        L.append(f"    min/max   : {np.min(arr):.4f} / {np.max(arr):.4f}")
        L.append(f"    p25/p75   : {np.percentile(arr,25):.4f} / {np.percentile(arr,75):.4f}")
        L.append(f"    p95/p99   : {np.percentile(arr,95):.4f} / {np.percentile(arr,99):.4f}")
        L.append(f"    >90% conf : {hc:,}  ({hc/len(arr)*100:.1f}%)")

    L.append(f"")
    L.append(SEP)
    L.append(f"  CONFIDENCE BUCKETS  (AI probability distribution across all files)")
    L.append(SEP)
    for lbl, n, pct in bucket_counts(all_ai, total):
        L.append(f"    {lbl:<28}  {n:>8,}  ({pct:5.1f}%)")

    rt = build_repo_table(rows)
    L.append(f"")
    L.append(SEP)
    L.append(f"  PER-REPO BREAKDOWN")
    L.append(SEP)
    L.append(f"  {'REPO':<35}  {'TOTAL':>7}  {'AI':>7}  {'HUMAN':>7}  {'AI%':>7}  {'MEAN_AI_CONF':>13}")
    L.append(f"  {'─'*90}")
    for repo, s in sorted(rt.items(), key=lambda x: -(x[1]["ai"] + x[1]["human"])):
        t = s["ai"] + s["human"]
        pct = s["ai"] / t * 100 if t else 0
        mc  = np.mean(s["ai_probs"]) if s["ai_probs"] else 0
        L.append(f"  {repo:<35}  {t:>7,}  {s['ai']:>7,}  {s['human']:>7,}  {pct:>6.1f}%  {mc:>13.4f}")

    return "\n".join(L)


def combined_analysis(domain, pre_rows, post_rows):
    pre_n  = len(pre_rows);  post_n  = len(post_rows)
    pre_ai  = sum(1 for r in pre_rows  if r["prediction"] == "AI")
    post_ai = sum(1 for r in post_rows if r["prediction"] == "AI")
    pre_pct  = pre_ai  / pre_n  * 100 if pre_n  else 0
    post_pct = post_ai / post_n * 100 if post_n else 0
    d_ppt    = post_pct - pre_pct
    rel_chg  = d_ppt / pre_pct * 100 if pre_pct else float("nan")

    pre_ap  = np.array([r["ai_confidence"] for r in pre_rows])
    post_ap = np.array([r["ai_confidence"] for r in post_rows])
    pre_rt  = build_repo_table(pre_rows)
    post_rt = build_repo_table(post_rows)
    common  = set(pre_rt) & set(post_rt)

    ks_stat, ks_p = ks_2samp(pre_ap, post_ap)
    mw_stat, mw_p = mannwhitneyu(pre_ap, post_ap, alternative="two-sided")
    pool_std = np.sqrt(
        (np.var(pre_ap) * (pre_n - 1) + np.var(post_ap) * (post_n - 1))
        / (pre_n + post_n - 2)
    ) if pre_n + post_n > 2 else 1
    cohens_d = (np.mean(post_ap) - np.mean(pre_ap)) / pool_std if pool_std else 0
    mag = ("negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5
           else "medium" if abs(cohens_d) < 0.8 else "large")

    repo_deltas = []
    for repo in common:
        p = pre_rt[repo]; q = post_rt[repo]
        pt = p["ai"] + p["human"]; qt = q["ai"] + q["human"]
        if pt < 3 or qt < 3:
            continue
        repo_deltas.append((q["ai"] / qt - p["ai"] / pt) * 100)
    repo_deltas = np.array(repo_deltas)

    SEP = "  " + "─" * 70
    L = []
    L.append(f"\n{'='*70}")
    L.append(f"  COMBINED ANALYSIS: {domain}  (Pre-LLM → Post-LLM)")
    L.append(f"{'='*70}")
    L.append(f"")
    L.append(f"  {'METRIC':<44}  {'PRE-LLM':>10}  {'POST-LLM':>10}  {'CHANGE':>12}")
    L.append(f"  {'─'*82}")
    L.append(f"  {'Total files':<44}  {pre_n:>10,}  {post_n:>10,}  {post_n-pre_n:>+12,}")
    L.append(f"  {'AI-detected files':<44}  {pre_ai:>10,}  {post_ai:>10,}  {post_ai-pre_ai:>+12,}")
    L.append(f"  {'AI detection rate':<44}  {pre_pct:>9.2f}%  {post_pct:>9.2f}%  {d_ppt:>+11.2f}pp")
    L.append(f"  {'Relative increase in AI rate':<44}  {'':>10}  {'':>10}  {rel_chg:>+11.2f}%")
    L.append(f"  {'Mean AI probability':<44}  {np.mean(pre_ap):>10.4f}  {np.mean(post_ap):>10.4f}  {np.mean(post_ap)-np.mean(pre_ap):>+12.4f}")
    L.append(f"  {'Median AI probability':<44}  {np.median(pre_ap):>10.4f}  {np.median(post_ap):>10.4f}  {np.median(post_ap)-np.median(pre_ap):>+12.4f}")
    L.append(f"  {'Std of AI probability':<44}  {np.std(pre_ap):>10.4f}  {np.std(post_ap):>10.4f}  {np.std(post_ap)-np.std(pre_ap):>+12.4f}")
    L.append(f"  {'Files > 90% AI confidence':<44}  {int(np.sum(pre_ap>0.9)):>10,}  {int(np.sum(post_ap>0.9)):>10,}  {int(np.sum(post_ap>0.9))-int(np.sum(pre_ap>0.9)):>+12,}")
    L.append(f"  {'% files > 90% AI confidence':<44}  {np.mean(pre_ap>0.9)*100:>9.2f}%  {np.mean(post_ap>0.9)*100:>9.2f}%  {(np.mean(post_ap>0.9)-np.mean(pre_ap>0.9))*100:>+11.2f}pp")
    L.append(f"")
    L.append(SEP)
    L.append(f"  STATISTICAL SIGNIFICANCE")
    L.append(SEP)
    L.append(f"  Kolmogorov-Smirnov: stat={ks_stat:.4f},  p={ks_p:.2e}"
             f"  →  {'SIGNIFICANT' if ks_p < 0.05 else 'not significant'} (α=0.05)")
    L.append(f"  Mann-Whitney U:     stat={mw_stat:.0f},  p={mw_p:.2e}"
             f"  →  {'SIGNIFICANT' if mw_p < 0.05 else 'not significant'} (α=0.05)")
    L.append(f"  Cohen's d:          {cohens_d:+.4f}  ({mag} effect)")
    L.append(f"  Direction:          {'AI probability INCREASED post-LLM' if cohens_d > 0 else 'AI probability DECREASED post-LLM'}")
    L.append(f"")
    L.append(SEP)
    L.append(f"  PER-REPO DELTA SUMMARY")
    L.append(SEP)
    L.append(f"  Repos compared          : {len(repo_deltas):,}")
    if len(repo_deltas):
        L.append(f"  Repos with AI rate ↑    : {int(np.sum(repo_deltas>0)):,}  ({np.mean(repo_deltas>0)*100:.1f}%)")
        L.append(f"  Repos with AI rate ↓    : {int(np.sum(repo_deltas<0)):,}  ({np.mean(repo_deltas<0)*100:.1f}%)")
        L.append(f"  Mean  Δ AI%             : {np.mean(repo_deltas):+.2f} pp")
        L.append(f"  Median Δ AI%            : {np.median(repo_deltas):+.2f} pp")
        L.append(f"  Largest increase        : {np.max(repo_deltas):+.2f} pp")
        L.append(f"  Largest decrease        : {np.min(repo_deltas):+.2f} pp")

    L.append(f"")
    L.append(SEP)
    L.append(f"  PER-REPO: PRE vs POST  (sorted by |Δ AI%|, min 3 files each)")
    L.append(SEP)
    L.append(f"  {'REPO':<35}  {'PRE_T':>6}  {'POST_T':>6}  {'PRE_AI%':>8}  {'POST_AI%':>9}  {'Δ pp':>8}  {'PRE_CONF':>9}  {'POST_CONF':>9}")
    L.append(f"  {'─'*106}")
    rows_out = []
    for repo in common:
        p = pre_rt[repo]; q = post_rt[repo]
        pt = p["ai"] + p["human"]; qt = q["ai"] + q["human"]
        if pt == 0 or qt == 0:
            continue
        pp = p["ai"] / pt * 100; qp = q["ai"] / qt * 100
        pc = np.mean(p["ai_probs"]); qc = np.mean(q["ai_probs"])
        rows_out.append((repo, pt, qt, pp, qp, qp - pp, pc, qc))
    for repo, pt, qt, pp, qp, dp, pc, qc in sorted(rows_out, key=lambda x: -abs(x[5])):
        L.append(f"  {repo:<35}  {pt:>6,}  {qt:>6,}  {pp:>7.1f}%  {qp:>8.1f}%  {dp:>+7.1f}pp  {pc:>9.4f}  {qc:>9.4f}")

    return "\n".join(L)


# ── visuals ───────────────────────────────────────────────────────────────────

def save_fig(fig, out_dir, name):
    fig.savefig(out_dir / name)
    plt.close(fig)


def plot_01_ai_rate(domain, pre_rows, post_rows, out_dir):
    """Clean before/after bar with delta arrow."""
    pre_n   = len(pre_rows);  post_n  = len(post_rows)
    pre_ai  = sum(1 for r in pre_rows  if r["prediction"] == "AI")
    post_ai = sum(1 for r in post_rows if r["prediction"] == "AI")
    pre_pct  = pre_ai  / pre_n  * 100
    post_pct = post_ai / post_n * 100
    d_ppt    = post_pct - pre_pct

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar([0, 1], [pre_pct, post_pct], width=0.5,
                  color=[PRE_C, POST_C], edgecolor="none", zorder=3)

    for bar, v in zip(bars, [pre_pct, post_pct]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=bar.get_facecolor())

    col = POST_C if d_ppt >= 0 else PRE_C
    ax.annotate("", xy=(1, post_pct), xytext=(0, pre_pct),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=20))
    sign = "▲" if d_ppt >= 0 else "▼"
    mid_y = (pre_pct + post_pct) / 2
    ax.text(0.5, mid_y + (3 if d_ppt >= 0 else -5),
            f"{sign} {abs(d_ppt):.2f} pp  ({d_ppt/pre_pct*100:+.1f}% relative)",
            ha="center", fontsize=11, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=col, linewidth=1.5))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-LLM Era", "Post-LLM Era"], fontsize=13)
    ax.set_ylabel("AI-Generated Code (%)", fontsize=11)
    ax.set_title(f"{domain}\nAI Detection Rate: Pre vs Post LLM Era",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(pre_pct, post_pct) * 1.3)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    ax.legend(handles=[
        mpatches.Patch(color=PRE_C,  label=f"Pre-LLM  ({pre_n:,} files)"),
        mpatches.Patch(color=POST_C, label=f"Post-LLM ({post_n:,} files)"),
    ], fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, out_dir, "01_ai_rate_comparison.png")


def plot_02_bubble_scatter(domain, pre_rows, post_rows, out_dir):
    """Bubble scatter: pre AI% vs post AI% per repo."""
    pre_rt  = build_repo_table(pre_rows)
    post_rt = build_repo_table(post_rows)
    common  = sorted(set(pre_rt) & set(post_rt))

    points = []
    for repo in common:
        p = pre_rt[repo]; q = post_rt[repo]
        pt = p["ai"] + p["human"]; qt = q["ai"] + q["human"]
        if pt < 3 or qt < 3:
            continue
        pp = p["ai"] / pt * 100; qp = q["ai"] / qt * 100
        points.append((repo, pp, qp, pt + qt))

    if not points:
        return
    repos, pre_p, post_p, sizes = zip(*points)
    pre_p  = np.array(pre_p);  post_p = np.array(post_p)
    sizes  = np.array(sizes,  dtype=float)
    deltas = post_p - pre_p
    s_sc   = 60 + (sizes / sizes.max()) * 700

    lim = max(pre_p.max(), post_p.max()) * 1.1

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1.2, alpha=0.55, zorder=1)
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color=POST_C, zorder=0)
    ax.fill_between([0, lim], [0, 0],   [0, lim],   alpha=0.04, color=PRE_C,  zorder=0)

    sc = ax.scatter(pre_p, post_p, s=s_sc, c=deltas,
                    cmap="RdYlGn_r", vmin=-40, vmax=40,
                    edgecolors="white", linewidths=0.6, alpha=0.88, zorder=3)

    top_idx = np.argsort(np.abs(deltas))[-8:]
    for i in top_idx:
        ax.annotate(repos[i], xy=(pre_p[i], post_p[i]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, alpha=0.85,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7, alpha=0.6))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Δ AI% (post − pre)", fontsize=10)
    ax.set_xlabel("Pre-LLM AI Detection Rate (%)", fontsize=11)
    ax.set_ylabel("Post-LLM AI Detection Rate (%)", fontsize=11)
    ax.set_title(f"{domain} — Per-Repository AI Rate Shift\n"
                 f"(bubble size ∝ total files;  above diagonal = AI increased post-LLM)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-2, lim); ax.set_ylim(-2, lim)
    ax.text(lim * 0.68, lim * 0.10, "AI decreased post-LLM",
            fontsize=8, color=PRE_C,  alpha=0.7, style="italic")
    ax.text(lim * 0.04, lim * 0.88, "AI increased post-LLM",
            fontsize=8, color=POST_C, alpha=0.7, style="italic")
    for sz_lbl, sz_v in [("Small", "80"), ("Medium", "300"), ("Large", "700")]:
        ax.scatter([], [], s=float(sz_v), c="gray", alpha=0.5,
                   edgecolors="white", label=sz_lbl)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, out_dir, "02_bubble_scatter.png")


def plot_03_distribution_shift(domain, pre_rows, post_rows, out_dir):
    """KDE showing shift in AI probability distribution."""
    pre_ap  = np.array([r["ai_confidence"] for r in pre_rows])
    post_ap = np.array([r["ai_confidence"] for r in post_rows])

    x = np.linspace(0, 1, 600)
    kde_pre  = gaussian_kde(pre_ap,  bw_method=0.07)(x)
    kde_post = gaussian_kde(post_ap, bw_method=0.07)(x)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(x, kde_pre,  alpha=0.25, color=PRE_C)
    ax.fill_between(x, kde_post, alpha=0.25, color=POST_C)
    ax.plot(x, kde_pre,  color=PRE_C,  lw=2.5, label=f"Pre-LLM  (mean={np.mean(pre_ap):.3f})")
    ax.plot(x, kde_post, color=POST_C, lw=2.5, label=f"Post-LLM (mean={np.mean(post_ap):.3f})")
    ax.fill_between(x, kde_pre, kde_post,
                    where=kde_post > kde_pre, alpha=0.35, color=POST_C, label="Post > Pre (AI↑)")
    ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.45, label="Decision boundary")
    ax.axvline(np.mean(pre_ap),  color=PRE_C,  lw=1.5, ls=":", alpha=0.8)
    ax.axvline(np.mean(post_ap), color=POST_C, lw=1.5, ls=":", alpha=0.8)

    ks_stat, ks_p = ks_2samp(pre_ap, post_ap)
    ax.text(0.35, 0.96, f"KS stat = {ks_stat:.4f},  p = {ks_p:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.set_xlabel("AI Probability Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{domain} — Shift in AI Probability Distribution\nPre-LLM Era vs Post-LLM Era",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    save_fig(fig, out_dir, "03_distribution_shift.png")


def plot_04_top_repos_delta(domain, pre_rows, post_rows, out_dir):
    """Horizontal bar: top repos by AI% change."""
    pre_rt  = build_repo_table(pre_rows)
    post_rt = build_repo_table(post_rows)
    common  = set(pre_rt) & set(post_rt)

    data = []
    for repo in common:
        p = pre_rt[repo]; q = post_rt[repo]
        pt = p["ai"] + p["human"]; qt = q["ai"] + q["human"]
        if pt < 5 or qt < 5:
            continue
        pp = p["ai"] / pt * 100; qp = q["ai"] / qt * 100
        data.append((repo, qp - pp, pp, qp, pt + qt))

    data.sort(key=lambda x: x[1])
    inc = [d for d in data if d[1] >= 0][-12:]
    dec = [d for d in data if d[1] < 0][:8]
    data = dec + inc
    if not data:
        return

    labels   = [d[0] for d in data]
    deltas   = [d[1] for d in data]
    pre_pct  = [d[2] for d in data]
    post_pct = [d[3] for d in data]
    colors   = [POST_C if d >= 0 else PRE_C for d in deltas]

    fig, ax = plt.subplots(figsize=(11, max(5, len(data) * 0.42)))
    y = np.arange(len(data))
    ax.barh(y, deltas, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5, height=0.65)
    ax.axvline(0, color="black", lw=1.2, zorder=3)

    for i, (dv, pv, qv) in enumerate(zip(deltas, pre_pct, post_pct)):
        pad = 0.4 if dv >= 0 else -0.4
        ha  = "left" if dv >= 0 else "right"
        ax.text(dv + pad, i, f"{pv:.0f}% → {qv:.0f}%",
                va="center", ha=ha, fontsize=7.8)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ AI Detection Rate (pp)", fontsize=11)
    ax.set_title(f"{domain} — Per-Repository Change in AI Detection Rate\n"
                 f"(Post-LLM minus Pre-LLM; min 5 files each split)",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=POST_C, label="AI rate increased post-LLM"),
        mpatches.Patch(color=PRE_C,  label="AI rate decreased post-LLM"),
    ], fontsize=9, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_dir, "04_top_repos_delta.png")


def plot_05_confidence_buckets(domain, pre_rows, post_rows, out_dir):
    """Grouped bar: confidence buckets pre vs post."""
    pre_ap  = np.array([r["ai_confidence"] for r in pre_rows])
    post_ap = np.array([r["ai_confidence"] for r in post_rows])

    buck_pre  = bucket_counts(pre_ap,  len(pre_ap))
    buck_post = bucket_counts(post_ap, len(post_ap))
    labels     = [b[0] for b in buck_pre]
    pre_pcts   = np.array([b[2] for b in buck_pre])
    post_pcts  = np.array([b[2] for b in buck_post])
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, pre_pcts,  width=w, color=PRE_C,  alpha=0.88,
           label="Pre-LLM",  edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, post_pcts, width=w, color=POST_C, alpha=0.88,
           label="Post-LLM", edgecolor="white", linewidth=0.5)

    for i, (pv, qv) in enumerate(zip(pre_pcts, post_pcts)):
        d = qv - pv
        if abs(d) >= 0.5:
            sign = "▲" if d > 0 else "▼"
            col  = POST_C if d > 0 else PRE_C
            ax.text(x[i] + w/2, qv + 0.5, f"{sign}{abs(d):.1f}pp",
                    ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of files in bucket", fontsize=11)
    ax.set_title(f"{domain} — Confidence Bucket Distribution: Pre vs Post LLM Era",
                 fontsize=12, fontweight="bold")
    ax.text(0.52, 0.97, "← Human dominant  |  AI dominant →",
            fontsize=8, color="gray", transform=ax.transAxes, va="top")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, "05_confidence_buckets.png")


# ── report writer ─────────────────────────────────────────────────────────────

def write_report(domain, pre_rows, post_rows, out_dir, pre_label, post_label):
    report_path = out_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  AI CODE DETECTION ANALYSIS REPORT\n")
        f.write(f"  Domain    : {domain}\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        f.write(stats_block(pre_rows,  pre_label))
        f.write("\n\n")
        f.write(stats_block(post_rows, post_label))
        f.write("\n\n")
        f.write(combined_analysis(domain, pre_rows, post_rows))
        f.write("\n")


# ── main ──────────────────────────────────────────────────────────────────────

def process_domain(domain, pre_path, post_path, search_dir):
    pre_rows  = load_csv(pre_path)
    post_rows = load_csv(post_path)
    if not pre_rows or not post_rows:
        return

    out_dir = search_dir / f"{domain}_summary_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_label  = pre_path.name.replace("_predictions.csv", "")
    post_label = post_path.name.replace("_predictions.csv", "")

    plot_01_ai_rate(domain, pre_rows, post_rows, out_dir)
    plot_02_bubble_scatter(domain, pre_rows, post_rows, out_dir)
    plot_03_distribution_shift(domain, pre_rows, post_rows, out_dir)
    plot_04_top_repos_delta(domain, pre_rows, post_rows, out_dir)
    plot_05_confidence_buckets(domain, pre_rows, post_rows, out_dir)

    write_report(domain, pre_rows, post_rows, out_dir, pre_label, post_label)


def main():
    _root = Path(__file__).resolve().parent
    search_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (_root / "results")
    pairs = find_pairs(search_dir)
    if not pairs:
        return
    for domain, pre_path, post_path in pairs:
        process_domain(domain, pre_path, post_path, search_dir)


if __name__ == "__main__":
    main()  