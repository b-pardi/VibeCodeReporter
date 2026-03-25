#!/usr/bin/env python3
"""
Visualization for AI code detection statistical analysis.

Reads stats_results.json (from stats/analyze.py) and prediction CSVs,
then writes publication-quality PNG figures to the output directory.

Figures produced:
  fig_metrics_bar.png        — Accuracy / F1 macro / F1(AI) / F1(human) with 95% CI
  fig_confusion_matrices.png — Confusion matrix per model (row-normalized + raw counts)
  fig_significance.png       — Pairwise McNemar p-value heatmap with Cohen's h
  fig_kappa.png              — Inter-model Cohen's kappa bar chart

Usage (Setting A):
    python stats/plot.py \\
        --results  ~/res/results/stats_settingA/stats_results.json \\
        --modernbert   ~/res/predictions/modernbert_diffs.csv \\
        --gptsniffer   ~/res/predictions/gptsniffer_code_settingA.csv \\
        --droiddetect  ~/res/predictions/droiddetect_zeroshot.csv \\
        --out          ~/res/results/stats_settingA/

Usage (Setting B — add fine-tuned DroidDetect):
    python stats/plot.py \\
        --results  ~/res/results/stats_settingB/stats_results.json \\
        --modernbert            ~/res/predictions/modernbert_diffs.csv \\
        --gptsniffer            ~/res/predictions/gptsniffer_code_settingB.csv \\
        --droiddetect           ~/res/predictions/droiddetect_zeroshot.csv \\
        --droiddetect-finetuned ~/res/predictions/droiddetect_finetuned.csv \\
        --out                   ~/res/results/stats_settingB/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
PALETTE = sns.color_palette('Set2')
LABEL_NAMES = ['AI-gen (0)', 'Human (1)']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(path):
    df = pd.read_csv(path)
    missing = {'sha', 'label', 'pred'} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def align_predictions(dfs_dict):
    """Inner-join all prediction DataFrames on 'sha'."""
    names = list(dfs_dict.keys())
    base = dfs_dict[names[0]][['sha', 'label']].copy()
    for name, df in dfs_dict.items():
        base = base.merge(
            df[['sha', 'pred']].rename(columns={'pred': f'pred_{name}'}),
            on='sha', how='inner')
    return base


# ---------------------------------------------------------------------------
# Figure 1: Metrics bar chart with 95% CI error bars
# ---------------------------------------------------------------------------

def plot_metrics(results, out_dir):
    models = results['models']
    names = list(models.keys())

    metrics_cfg = [
        ('accuracy',  'Accuracy'),
        ('f1_macro',  'F1 Macro'),
        ('f1_ai',     'F1 (AI)'),
        ('f1_human',  'F1 (Human)'),
    ]

    n_m = len(metrics_cfg)
    n_models = len(names)
    x = np.arange(n_m)
    width = 0.7 / n_models
    offsets = np.linspace(-(n_models - 1) / 2 * width,
                           (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        m = models[name]
        points = [m[k]['point']            for k, _ in metrics_cfg]
        lo_err = [m[k]['point'] - m[k]['lo'] for k, _ in metrics_cfg]
        hi_err = [m[k]['hi'] - m[k]['point'] for k, _ in metrics_cfg]
        ax.bar(x + offsets[i], points, width,
               label=name, color=PALETTE[i % len(PALETTE)],
               yerr=np.array([lo_err, hi_err]), capsize=4,
               error_kw=dict(elinewidth=1.2, alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics_cfg])
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
              ncols=n_models, fontsize=9, framealpha=0.9)
    ax.set_title('Model Performance with 95% Bootstrap CI', pad=40)
    fig.tight_layout()

    path = out_dir / 'fig_metrics_bar.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Confusion matrices (row-normalised + raw counts)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(aligned, names, out_dir):
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    y_true = aligned['label'].to_numpy()
    for ax, name in zip(axes, names):
        y_pred = aligned[f'pred_{name}'].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Annotation: "83.2%\n(n=412)"
        annot = np.array([
            [f'{cm_norm[r, c]:.1%}\n(n={cm[r, c]})'
             for c in range(2)] for r in range(2)
        ])

        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                    ax=ax, cbar=False, vmin=0, vmax=1,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 10})
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    fig.suptitle('Confusion Matrices (row-normalised)', y=1.01)
    fig.tight_layout()

    path = out_dir / 'fig_confusion_matrices.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: McNemar p-value heatmap
# ---------------------------------------------------------------------------

def plot_significance(results, out_dir):
    names = list(results['models'].keys())
    n = len(names)
    name_to_i = {name: i for i, name in enumerate(names)}

    p_matrix = np.ones((n, n))
    h_matrix = np.full((n, n), np.nan)

    for pair in results['pairwise_tests']:
        a, b = pair['pair'].split(' vs ', 1)
        i, j = name_to_i.get(a), name_to_i.get(b)
        if i is None or j is None:
            continue
        p_matrix[i, j] = pair['p_holm']
        p_matrix[j, i] = pair['p_holm']
        h_matrix[i, j] = pair['cohen_h']
        h_matrix[j, i] = -pair['cohen_h']

    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = '—'
            else:
                p = p_matrix[i, j]
                stars = ('***' if p < 0.001 else
                         '**'  if p < 0.01  else
                         '*'   if p < 0.05  else 'ns')
                h = h_matrix[i, j]
                h_str = f'\nh={h:+.2f}' if not np.isnan(h) else ''
                annot[i, j] = f'p={p:.3f}\n{stars}{h_str}'

    mask = np.eye(n, dtype=bool)
    cell = max(2.0, 1.8 * n)
    fig, ax = plt.subplots(figsize=(cell * n, cell * (n - 0.2)))
    sns.heatmap(p_matrix, mask=mask, annot=annot, fmt='',
                cmap='RdYlGn_r', vmin=0, vmax=0.1,
                xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'p-value (Holm-Bonferroni)'})

    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                   fill=True, color='#e0e0e0', zorder=2))
        ax.text(i + 0.5, i + 0.5, '—',
                ha='center', va='center', fontsize=12, zorder=3)

    ax.set_title("Pairwise McNemar's Test (Holm-Bonferroni corrected)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()

    path = out_dir / 'fig_significance.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Inter-model Cohen's kappa bar chart
# ---------------------------------------------------------------------------

def plot_kappa(results, out_dir):
    kappas = results['inter_model_kappas']
    if not kappas:
        print("  Skipping kappa plot — no inter-model kappas in JSON.")
        return

    pairs  = list(kappas.keys())
    values = [kappas[p] for p in pairs]

    fig, ax = plt.subplots(figsize=(max(5, len(pairs) * 1.8), 4.5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(pairs))]
    bars = ax.bar(pairs, values, color=colors, width=0.5)

    ax.set_ylabel("Cohen's κ")
    ax.set_ylim(-1, 1.1)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Interpretation reference lines
    for y, label in [
        (0.20, 'fair'),
        (0.40, 'moderate'),
        (0.60, 'substantial'),
        (0.80, 'near-perfect'),
    ]:
        ax.axhline(y, color='gray', linewidth=0.5, linestyle=':')
        ax.text(len(pairs) - 0.55, y + 0.015, label,
                ha='right', va='bottom', fontsize=7, color='gray')

    ax.set_xticklabels(pairs, rotation=20, ha='right', fontsize=9)
    ax.set_title("Inter-Model Prediction Agreement (Cohen's κ)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.02 if val >= 0 else -0.04),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()

    path = out_dir / 'fig_kappa.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from stats/analyze.py results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results', required=True,
                        help='Path to stats_results.json produced by analyze.py')
    parser.add_argument('--modernbert', required=True,
                        help='Predictions CSV for ModernBERT (diff-trained)')
    parser.add_argument('--gptsniffer', required=True,
                        help='Predictions CSV for GPTSniffer (code-trained)')
    parser.add_argument('--droiddetect', required=True,
                        help='Predictions CSV for DroidDetect zero-shot')
    parser.add_argument('--droiddetect-finetuned', default=None,
                        help='Predictions CSV for DroidDetect fine-tuned (Setting B, optional)')
    parser.add_argument('--out', default=None,
                        help='Output directory for PNGs (default: same directory as --results)')
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out) if args.out else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text())

    # Build model → CSV mapping.  Keys must match the display names used in
    # analyze.py so they align with the JSON results.
    csv_map = {
        'ModernBERT (diffs)': args.modernbert,
        'GPTSniffer (code)':  args.gptsniffer,
        'DroidDetect (code)': args.droiddetect,
    }
    if args.droiddetect_finetuned:
        csv_map['DroidDetect fine-tuned (code)'] = args.droiddetect_finetuned

    # Only keep models that appear in the JSON (handles partial runs)
    valid_names = [n for n in csv_map if n in results['models']]
    if not valid_names:
        raise SystemExit("No model names in the JSON match the expected keys. "
                         "Check that --results points to the right JSON file.")

    print("Loading and aligning predictions ...")
    dfs = {name: load_predictions(csv_map[name]) for name in valid_names}
    aligned = align_predictions(dfs)
    print(f"  Aligned rows: {len(aligned)}")

    print("\nGenerating figures ...")
    plot_metrics(results, out_dir)
    plot_confusion_matrices(aligned, valid_names, out_dir)
    plot_significance(results, out_dir)
    plot_kappa(results, out_dir)

    print(f"\nAll figures written to: {out_dir}")
    print("  fig_metrics_bar.png        — accuracy / F1 with 95% CI")
    print("  fig_confusion_matrices.png — per-model confusion matrices")
    print("  fig_significance.png       — McNemar p-value heatmap")
    print("  fig_kappa.png              — inter-model Cohen's kappa")


if __name__ == '__main__':
    main()
