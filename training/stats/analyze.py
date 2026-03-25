#!/usr/bin/env python3
"""
Statistical comparison of AI code detection models.

Reads prediction CSVs (sha, repo, date, label, pred) from multiple models,
aligns them on sha, then computes:
  - Bootstrap confidence intervals (95%, n=10000) for accuracy, F1, precision, recall
  - McNemar's test for pairwise model significance (same test set)
  - Holm-Bonferroni correction across 3 pairwise comparisons
  - Cohen's h (effect size for proportion differences)
  - Cohen's kappa (inter-rater agreement)

Outputs a Markdown results table and a JSON file.

Usage:
    # Minimum (in-distribution parquet eval):
    python stats/analyze.py \\
        --modernbert  predictions/modernbert_diffs.csv \\
        --gptsniffer  predictions/gptsniffer_code.csv \\
        --droiddetect predictions/droiddetect_code.csv \\
        --out         results/

    # Full (also include Setting B fine-tuned DroidDetect):
    python stats/analyze.py \\
        --modernbert           predictions/modernbert_diffs.csv \\
        --gptsniffer           predictions/gptsniffer_code.csv \\
        --droiddetect          predictions/droiddetect_zeroshot.csv \\
        --droiddetect-finetuned predictions/droiddetect_finetuned.csv \\
        --out                  results/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, classification_report,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=10000, ci=0.95, seed=42):
    """Percentile bootstrap CI for any scalar metric.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            stats.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            stats.append(float('nan'))
    stats = np.array(stats)
    stats = stats[~np.isnan(stats)]
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 - (1 - ci) / 2) * 100)
    return float(metric_fn(y_true, y_pred)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, preds_a, preds_b):
    """McNemar's test comparing model A vs model B on the same test set.

    Returns (chi2_statistic, p_value). Uses continuity correction.
    Both correct (n11), A correct B wrong (b), A wrong B correct (c), both wrong (n00).
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    n11 = int(np.sum(correct_a  & correct_b))   # both correct
    b   = int(np.sum(correct_a  & ~correct_b))  # A correct, B wrong
    c   = int(np.sum(~correct_a & correct_b))   # A wrong, B correct
    n00 = int(np.sum(~correct_a & ~correct_b))  # both wrong
    table = [[n11, b], [c, n00]]
    if b + c < 20:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False, correction=True)
    return float(result.statistic), float(result.pvalue), b, c


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohen_h(p1, p2):
    """Cohen's h: effect size for difference between two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def interpret_h(h):
    ah = abs(h)
    if ah < 0.2:
        return 'small'
    elif ah < 0.5:
        return 'medium'
    else:
        return 'large'


# ---------------------------------------------------------------------------
# Per-model metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true, y_pred, n_bootstrap=10000, seed=42):
    """Compute all metrics with bootstrap CIs for one model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}

    def acc_fn(yt, yp):     return accuracy_score(yt, yp)
    def f1_mac(yt, yp):     return f1_score(yt, yp, average='macro', zero_division=0)
    def f1_ai(yt, yp):      return f1_score(yt, yp, pos_label=0, average='binary', zero_division=0)
    def f1_human(yt, yp):   return f1_score(yt, yp, pos_label=1, average='binary', zero_division=0)
    def prec_ai(yt, yp):    return precision_score(yt, yp, pos_label=0, average='binary', zero_division=0)
    def prec_human(yt, yp): return precision_score(yt, yp, pos_label=1, average='binary', zero_division=0)
    def rec_ai(yt, yp):     return recall_score(yt, yp, pos_label=0, average='binary', zero_division=0)
    def rec_human(yt, yp):  return recall_score(yt, yp, pos_label=1, average='binary', zero_division=0)

    for name, fn in tqdm([
        ('accuracy',      acc_fn),
        ('f1_macro',      f1_mac),
        ('f1_ai',         f1_ai),
        ('f1_human',      f1_human),
        ('precision_ai',  prec_ai),
        ('precision_human', prec_human),
        ('recall_ai',     rec_ai),
        ('recall_human',  rec_human),
    ], desc="Bootstrapping metrics", leave=False):
        pt, lo, hi = bootstrap_ci(y_true, y_pred, fn,
                                  n_bootstrap=n_bootstrap, seed=seed)
        metrics[name] = {'point': pt, 'lo': lo, 'hi': hi}

    metrics['kappa_vs_truth'] = float(cohen_kappa_score(y_true, y_pred))
    metrics['n'] = int(len(y_true))
    return metrics


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_ci(m, key):
    d = m[key]
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def fmt_h(h):
    return f"{h:+.3f} ({interpret_h(h)})"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def load_predictions(path):
    """Load predictions CSV (sha, repo, date, label, pred).

    Deduplicates by SHA, keeping the first occurrence per SHA.
    Duplicate SHAs arise when the same commit exists in multiple cloned repos
    (fork scenario). Deduplication ensures McNemar's test sees each commit once,
    preserving the independence assumption.
    """
    df = pd.read_csv(path)
    required = {'sha', 'label', 'pred'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    n_before = len(df)
    df = df.drop_duplicates(subset='sha', keep='first').reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Deduplicated {path}: {n_before} → {len(df)} rows "
              f"({n_dropped} duplicate SHAs removed)", file=sys.stderr)
    return df


def align_predictions(*dfs):
    """Inner-join multiple prediction DataFrames on 'sha'."""
    base = dfs[0][['sha', 'label']].copy()
    for i, df in enumerate(dfs):
        base = base.merge(
            df[['sha', 'pred']].rename(columns={'pred': f'pred_{i}'}),
            on='sha', how='inner')
    if len(base) < len(dfs[0]):
        n_dropped = len(dfs[0]) - len(base)
        print(f"WARNING: {n_dropped} rows dropped during sha alignment "
              f"({len(base)} rows retained).", file=sys.stderr)
    return base


def run_analysis(models, out_dir, n_bootstrap=10000, seed=42):
    """Run full statistical analysis.

    Args:
        models: dict of {display_name: predictions_csv_path}
        out_dir: Path to write results
        n_bootstrap: Bootstrap iterations
        seed: Random seed
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions ...", file=sys.stderr)
    dfs = {}
    for name, path in models.items():
        print(f"  {name}: {path}", file=sys.stderr)
        dfs[name] = load_predictions(path)

    # Align on sha
    print("Aligning on sha ...", file=sys.stderr)
    names = list(dfs.keys())
    aligned = align_predictions(*[dfs[n] for n in names])
    y_true = aligned['label'].tolist()
    print(f"  Aligned rows: {len(aligned)}", file=sys.stderr)

    # Per-model metrics
    print(f"Computing metrics (n_bootstrap={n_bootstrap}) ...", file=sys.stderr)
    all_metrics = {}
    all_preds = {}
    for i, name in enumerate(tqdm(names, desc="Models")):
        y_pred = aligned[f'pred_{i}'].tolist()
        all_preds[name] = y_pred
        print(f"  {name} ...", file=sys.stderr)
        all_metrics[name] = compute_all_metrics(y_true, y_pred,
                                                n_bootstrap=n_bootstrap, seed=seed)
        # Full classification report
        print(f"\n{name}:")
        print(classification_report(y_true, y_pred,
                                    target_names=['AI-generated(0)', 'Human-written(1)'],
                                    zero_division=0))

    # Pairwise McNemar + Cohen's h
    print("\nPairwise significance tests ...", file=sys.stderr)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            chi2, p_raw, b, c = mcnemar_test(y_true, all_preds[na], all_preds[nb])
            acc_a = all_metrics[na]['accuracy']['point']
            acc_b = all_metrics[nb]['accuracy']['point']
            h = cohen_h(acc_a, acc_b)
            pairs.append({
                'pair': f'{na} vs {nb}',
                'chi2': chi2, 'p_raw': p_raw, 'b': b, 'c': c,
                'cohen_h': h, 'cohen_h_interp': interpret_h(h),
            })

    # Holm-Bonferroni correction
    p_values = [p['p_raw'] for p in pairs]
    if len(p_values) > 1:
        reject, p_corr, _, _ = multipletests(p_values, method='holm')
    else:
        reject = [p_values[0] < 0.05]
        p_corr = p_values
    for i, pair in enumerate(pairs):
        pair['p_holm'] = float(p_corr[i])
        pair['reject_h0'] = bool(reject[i])

    # Inter-model kappas
    print("\nInter-model kappas ...", file=sys.stderr)
    kappas = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            k = cohen_kappa_score(all_preds[na], all_preds[nb])
            kappas[f'{na} vs {nb}'] = float(k)

    # ---------------------------------------------------------------------------
    # Build output
    # ---------------------------------------------------------------------------

    # Table 1: Per-model metrics
    table1_header = (
        f"| {'Model':<35} | {'Accuracy':^24} | {'F1 macro':^24} | "
        f"{'F1 (AI)':^24} | {'F1 (human)':^24} | Kappa vs truth |"
    )
    table1_sep = '|' + '-' * 37 + '|' + ('-' * 26 + '|') * 4 + '-' * 16 + '|'
    table1_rows = [table1_header, table1_sep]
    for name in names:
        m = all_metrics[name]
        row = (
            f"| {name:<35} | {fmt_ci(m, 'accuracy'):^24} | "
            f"{fmt_ci(m, 'f1_macro'):^24} | {fmt_ci(m, 'f1_ai'):^24} | "
            f"{fmt_ci(m, 'f1_human'):^24} | {m['kappa_vs_truth']:.4f}         |"
        )
        table1_rows.append(row)

    # Table 2: Pairwise significance
    table2_header = (
        f"| {'Pair':<45} | {'chi2':>7} | {'p_raw':>8} | {'p_holm':>8} | "
        f"{'reject':>6} | {'Cohen h':^14} |"
    )
    table2_sep = '|' + '-' * 47 + '|' + ('-' * 9 + '|') * 3 + '-' * 8 + '|' + '-' * 16 + '|'
    table2_rows = [table2_header, table2_sep]
    for p in pairs:
        row = (
            f"| {p['pair']:<45} | {p['chi2']:>7.2f} | {p['p_raw']:>8.4f} | "
            f"{p['p_holm']:>8.4f} | {str(p['reject_h0']):>6} | "
            f"{fmt_h(p['cohen_h']):^14} |"
        )
        table2_rows.append(row)

    # Table 3: Inter-model kappas
    table3_header = f"| {'Pair':<45} | {'Kappa':>8} |"
    table3_sep = '|' + '-' * 47 + '|' + '-' * 10 + '|'
    table3_rows = [table3_header, table3_sep]
    for pair_name, k in kappas.items():
        table3_rows.append(f"| {pair_name:<45} | {k:>8.4f} |")

    md_lines = [
        "# AI Code Detection: Statistical Comparison",
        "",
        f"**Test set size:** {len(aligned)} samples  ",
        f"**Bootstrap iterations:** {n_bootstrap}  ",
        f"**Confidence level:** 95%  ",
        f"**Multiple testing correction:** Holm-Bonferroni  ",
        "",
        "## Table 1: Per-Model Performance (mean [95% CI])",
        "",
        *table1_rows,
        "",
        "## Table 2: Pairwise McNemar's Test",
        "",
        "> `b` = model A correct & B wrong, `c` = A wrong & B correct  ",
        "> p_holm = Holm-Bonferroni corrected p-value  ",
        "",
        *table2_rows,
        "",
        "## Table 3: Inter-Model Cohen's Kappa",
        "",
        *table3_rows,
        "",
        "## Notes",
        "- Label convention: 0=AI-generated, 1=Human-written",
        "- McNemar's test uses continuity correction (or exact when b+c < 20)",
        "- Cohen's h: |h| < 0.2 = small, 0.2–0.5 = medium, > 0.5 = large",
    ]

    md_path = out_dir / 'stats_results.md'
    md_path.write_text('\n'.join(md_lines))
    print(f"\nMarkdown results: {md_path}")

    json_results = {
        'n_samples': len(aligned),
        'n_bootstrap': n_bootstrap,
        'seed': seed,
        'models': {name: all_metrics[name] for name in names},
        'pairwise_tests': pairs,
        'inter_model_kappas': kappas,
    }
    json_path = out_dir / 'stats_results.json'
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"JSON results:     {json_path}")

    # Print tables to stdout
    print("\n" + "=" * 80)
    print('\n'.join(md_lines))

    return json_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Statistical comparison of AI code detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--modernbert', required=True,
                        help='Predictions CSV for ModernBERT (diff-trained)')
    parser.add_argument('--gptsniffer', required=True,
                        help='Predictions CSV for GPTSniffer (code-trained)')
    parser.add_argument('--droiddetect', required=True,
                        help='Predictions CSV for DroidDetect zero-shot or fine-tuned')
    parser.add_argument('--droiddetect-finetuned', default=None,
                        help='Optional: predictions CSV for DroidDetect fine-tuned (Setting B)')
    parser.add_argument('--out', default='results/',
                        help='Output directory for Markdown and JSON results (default: results/)')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Bootstrap iterations (default: 10000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    models = {
        'ModernBERT (diffs)':    args.modernbert,
        'GPTSniffer (code)':     args.gptsniffer,
        'DroidDetect (code)':    args.droiddetect,
    }
    if args.droiddetect_finetuned:
        models['DroidDetect fine-tuned (code)'] = args.droiddetect_finetuned

    run_analysis(models, args.out, n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == '__main__':
    main()
