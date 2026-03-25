#!/usr/bin/env bash
# =============================================================================
# VibeCodeReporter: Analysis Runner (Phases 4–6)
# =============================================================================
# Runs in-distribution eval, OOD eval, and statistical analysis.
# Training (Phases 1–3) must be completed first.
#
# Required model dirs (must contain a final_model/ subdirectory):
#   MODERNBERT_OUT   ~/res/modernbert_diffs_output/final_model
#   GPTSNIFFER_A_OUT ~/res/gptsniffer_hvai_output/final_model   (Setting A)
#   GPTSNIFFER_B_OUT ~/res/gptsniffer_code_output/final_model   (Setting B)
#   DROIDDETECT_OUT  ~/res/droiddetect_output/final_model       (Setting B)
#
# Required parquets (produced by collect.py export + code-export):
#   ~/data/git-data/ghcommits-export/test.parquet
#   ~/data/git-data/ghcommits-export/test_code.parquet
#
# Required OOD data (optional — skipped if not present):
#   ~/data/git-data/humanvsaicode_java/CONF/{training_data,testing_data}/
#
# Usage (run from training/):
#   bash run_analysis.sh                          # full run
#   bash run_analysis.sh --smoke-test             # 1000-sample sanity check
#   bash run_analysis.sh --resume                 # skip steps whose output already exists
#   bash run_analysis.sh --resume --smoke-test    # both
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit if your layout differs
# ---------------------------------------------------------------------------
EXPORT_DIR="$HOME/data/git-data/ghcommits-export"
HVAI_DIR="$HOME/data/git-data/humanvsaicode_java"

MODERNBERT_OUT="$HOME/res/modernbert_diffs_output"
GPTSNIFFER_A_OUT="$HOME/res/gptsniffer_hvai_output"
GPTSNIFFER_B_OUT="$HOME/res/gptsniffer_code_output"
DROIDDETECT_OUT="$HOME/res/droiddetect_output"

PREDS_DIR="$HOME/res/predictions"
RESULTS_DIR="$HOME/res/results"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
MAX_SAMPLES=""
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --smoke-test) MAX_SAMPLES="--max-samples 1000" ;;
        --resume)     RESUME=1 ;;
    esac
done
[[ "$MAX_SAMPLES" != "" ]] && echo "=== SMOKE TEST MODE (max 1000 samples) ==="
[[ "$RESUME" -eq 1 ]]      && echo "=== RESUME MODE (skipping completed steps) ==="

mkdir -p "$PREDS_DIR" "$RESULTS_DIR"

# Helper: skip a step if the final_model dir is missing
require_model() {
    local label="$1" dir="$2"
    if [[ ! -d "$dir" ]]; then
        echo "  WARNING: $label final_model not found at $dir — skipping."
        return 1
    fi
    return 0
}

# Helper: in --resume mode, skip a step if its output file/dir already exists
skip_if_done() {
    local label="$1" output="$2"
    if [[ "$RESUME" -eq 1 && ( -f "$output" || -d "$output" ) ]]; then
        echo "  [resume] $label — already done, skipping."
        return 0   # signal: skip
    fi
    return 1       # signal: run
}

# ---------------------------------------------------------------------------
# Phase 4 — In-Distribution Eval (test parquets → prediction CSVs)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 4: In-distribution eval ==="

if require_model "ModernBERT" "$MODERNBERT_OUT/final_model" && \
   ! skip_if_done "ModernBERT eval-diffs" "$PREDS_DIR/modernbert_diffs.csv"; then
    echo "  ModernBERT (diffs) on test.parquet ..."
    python modernbert.py eval-diffs \
        --model-dir  "$MODERNBERT_OUT/final_model" \
        --diffs      "$EXPORT_DIR/test.parquet" \
        --save-preds "$PREDS_DIR/modernbert_diffs.csv" \
        $MAX_SAMPLES
fi

if require_model "GPTSniffer Setting A" "$GPTSNIFFER_A_OUT/final_model" && \
   ! skip_if_done "GPTSniffer Setting A eval-diffs" "$PREDS_DIR/gptsniffer_code_settingA.csv"; then
    echo "  GPTSniffer (Setting A — HumanVsAICode-trained) on test_code.parquet ..."
    python gptsniffer.py eval-diffs \
        --model-dir  "$GPTSNIFFER_A_OUT/final_model" \
        --diffs      "$EXPORT_DIR/test_code.parquet" \
        --save-preds "$PREDS_DIR/gptsniffer_code_settingA.csv" \
        $MAX_SAMPLES
fi

if require_model "GPTSniffer Setting B" "$GPTSNIFFER_B_OUT/final_model" && \
   ! skip_if_done "GPTSniffer Setting B eval-diffs" "$PREDS_DIR/gptsniffer_code_settingB.csv"; then
    echo "  GPTSniffer (Setting B — code-parquet-trained) on test_code.parquet ..."
    python gptsniffer.py eval-diffs \
        --model-dir  "$GPTSNIFFER_B_OUT/final_model" \
        --diffs      "$EXPORT_DIR/test_code.parquet" \
        --save-preds "$PREDS_DIR/gptsniffer_code_settingB.csv" \
        $MAX_SAMPLES
fi

if ! skip_if_done "DroidDetect zero-shot eval-diffs" "$PREDS_DIR/droiddetect_zeroshot.csv"; then
    echo "  DroidDetect zero-shot on test_code.parquet ..."
    python droiddetect.py eval-diffs \
        --diffs      "$EXPORT_DIR/test_code.parquet" \
        --save-preds "$PREDS_DIR/droiddetect_zeroshot.csv" \
        $MAX_SAMPLES
fi

if require_model "DroidDetect fine-tuned" "$DROIDDETECT_OUT/final_model" && \
   ! skip_if_done "DroidDetect fine-tuned eval-diffs" "$PREDS_DIR/droiddetect_finetuned.csv"; then
    echo "  DroidDetect fine-tuned (Setting B) on test_code.parquet ..."
    python droiddetect.py eval-diffs \
        --model-dir  "$DROIDDETECT_OUT/final_model" \
        --diffs      "$EXPORT_DIR/test_code.parquet" \
        --save-preds "$PREDS_DIR/droiddetect_finetuned.csv" \
        $MAX_SAMPLES
fi

# ---------------------------------------------------------------------------
# Phase 5 — OOD Eval
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 5: OOD eval ==="

# HumanVsAICode OOD (skip if dataset not present)
if [[ -d "$HVAI_DIR/CONF/testing_data" ]]; then
    echo "  -- HumanVsAICode --"

    if require_model "ModernBERT" "$MODERNBERT_OUT/final_model" && \
       ! skip_if_done "ModernBERT eval-test" "$RESULTS_DIR/modernbert_hvai.txt"; then
        python modernbert.py eval-test \
            --model-dir "$MODERNBERT_OUT/final_model" \
            --data-dir  "$HVAI_DIR" \
            $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/modernbert_hvai.txt"
    fi

    if require_model "GPTSniffer Setting A" "$GPTSNIFFER_A_OUT/final_model" && \
       ! skip_if_done "GPTSniffer A eval-test" "$RESULTS_DIR/gptsniffer_a_hvai.txt"; then
        python gptsniffer.py eval-test \
            --model-dir "$GPTSNIFFER_A_OUT/final_model" \
            --data-dir  "$HVAI_DIR" \
            $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/gptsniffer_a_hvai.txt"
    fi

    if require_model "GPTSniffer Setting B" "$GPTSNIFFER_B_OUT/final_model" && \
       ! skip_if_done "GPTSniffer B eval-test" "$RESULTS_DIR/gptsniffer_b_hvai.txt"; then
        python gptsniffer.py eval-test \
            --model-dir "$GPTSNIFFER_B_OUT/final_model" \
            --data-dir  "$HVAI_DIR" \
            $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/gptsniffer_b_hvai.txt"
    fi

    if ! skip_if_done "DroidDetect zero-shot eval-test" "$RESULTS_DIR/droiddetect_zeroshot_hvai.txt"; then
        python droiddetect.py eval-test \
            --data-dir "$HVAI_DIR" \
            $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/droiddetect_zeroshot_hvai.txt"
    fi

    if require_model "DroidDetect fine-tuned" "$DROIDDETECT_OUT/final_model" && \
       ! skip_if_done "DroidDetect fine-tuned eval-test" "$RESULTS_DIR/droiddetect_finetuned_hvai.txt"; then
        python droiddetect.py eval-test \
            --model-dir "$DROIDDETECT_OUT/final_model" \
            --data-dir  "$HVAI_DIR" \
            $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/droiddetect_finetuned_hvai.txt"
    fi
else
    echo "  HumanVsAICode data not found at $HVAI_DIR — skipping OOD eval."
    echo "  Download with: git clone --sparse https://github.com/ASE2023-GPTSniffer/GPTSniffer.git"
fi

# Daniotti OOD (skip if parquet not present)
DANIOTTI_PARQUET="data/final_data_2/pyfunctions_ai_classified.parquet"
if [[ -f "$DANIOTTI_PARQUET" ]]; then
    echo "  -- Daniotti --"

    if require_model "ModernBERT" "$MODERNBERT_OUT/final_model" && \
       ! skip_if_done "ModernBERT eval-daniotti" "$RESULTS_DIR/modernbert_daniotti.txt"; then
        python modernbert.py eval-daniotti \
            --model-dir "$MODERNBERT_OUT/final_model" \
            2>&1 | tee "$RESULTS_DIR/modernbert_daniotti.txt"
    fi

    if require_model "GPTSniffer Setting A" "$GPTSNIFFER_A_OUT/final_model" && \
       ! skip_if_done "GPTSniffer A eval-daniotti" "$RESULTS_DIR/gptsniffer_a_daniotti.txt"; then
        python gptsniffer.py eval-daniotti \
            --model-dir "$GPTSNIFFER_A_OUT/final_model" \
            2>&1 | tee "$RESULTS_DIR/gptsniffer_a_daniotti.txt"
    fi

    if require_model "GPTSniffer Setting B" "$GPTSNIFFER_B_OUT/final_model" && \
       ! skip_if_done "GPTSniffer B eval-daniotti" "$RESULTS_DIR/gptsniffer_b_daniotti.txt"; then
        python gptsniffer.py eval-daniotti \
            --model-dir "$GPTSNIFFER_B_OUT/final_model" \
            2>&1 | tee "$RESULTS_DIR/gptsniffer_b_daniotti.txt"
    fi

    if ! skip_if_done "DroidDetect zero-shot eval-daniotti" "$RESULTS_DIR/droiddetect_zeroshot_daniotti.txt"; then
        python droiddetect.py eval-daniotti \
            2>&1 | tee "$RESULTS_DIR/droiddetect_zeroshot_daniotti.txt"
    fi

    if require_model "DroidDetect fine-tuned" "$DROIDDETECT_OUT/final_model" && \
       ! skip_if_done "DroidDetect fine-tuned eval-daniotti" "$RESULTS_DIR/droiddetect_finetuned_daniotti.txt"; then
        python droiddetect.py eval-daniotti \
            --model-dir "$DROIDDETECT_OUT/final_model" \
            2>&1 | tee "$RESULTS_DIR/droiddetect_finetuned_daniotti.txt"
    fi
else
    echo "  Daniotti parquet not found at $DANIOTTI_PARQUET — skipping."
fi

# ---------------------------------------------------------------------------
# Phase 6 — Statistical Analysis
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 6: Statistical analysis ==="

# Setting A: zero-shot/original-training baselines vs diff-trained ModernBERT
if [[ -f "$PREDS_DIR/modernbert_diffs.csv" && \
      -f "$PREDS_DIR/gptsniffer_code_settingA.csv" && \
      -f "$PREDS_DIR/droiddetect_zeroshot.csv" ]] && \
   ! skip_if_done "stats Setting A" "$RESULTS_DIR/stats_settingA.txt"; then
    echo "  Setting A ..."
    python stats/analyze.py \
        --modernbert  "$PREDS_DIR/modernbert_diffs.csv" \
        --gptsniffer  "$PREDS_DIR/gptsniffer_code_settingA.csv" \
        --droiddetect "$PREDS_DIR/droiddetect_zeroshot.csv" \
        --out         "$RESULTS_DIR/stats_settingA/" \
        2>&1 | tee "$RESULTS_DIR/stats_settingA.txt"
else
    [[ ! -f "$PREDS_DIR/modernbert_diffs.csv" || \
       ! -f "$PREDS_DIR/gptsniffer_code_settingA.csv" || \
       ! -f "$PREDS_DIR/droiddetect_zeroshot.csv" ]] && \
        echo "  Setting A prediction CSVs not all present — skipping." || true
fi

# Setting B: all three trained on same data, only modality differs
if [[ -f "$PREDS_DIR/modernbert_diffs.csv" && \
      -f "$PREDS_DIR/gptsniffer_code_settingB.csv" && \
      -f "$PREDS_DIR/droiddetect_zeroshot.csv" ]] && \
   ! skip_if_done "stats Setting B" "$RESULTS_DIR/stats_settingB.txt"; then
    echo "  Setting B ..."
    EXTRA_B=""
    if [[ -f "$PREDS_DIR/droiddetect_finetuned.csv" ]]; then
        EXTRA_B="--droiddetect-finetuned $PREDS_DIR/droiddetect_finetuned.csv"
    fi
    python stats/analyze.py \
        --modernbert  "$PREDS_DIR/modernbert_diffs.csv" \
        --gptsniffer  "$PREDS_DIR/gptsniffer_code_settingB.csv" \
        --droiddetect "$PREDS_DIR/droiddetect_zeroshot.csv" \
        $EXTRA_B \
        --out         "$RESULTS_DIR/stats_settingB/" \
        2>&1 | tee "$RESULTS_DIR/stats_settingB.txt"
else
    [[ ! -f "$PREDS_DIR/modernbert_diffs.csv" || \
       ! -f "$PREDS_DIR/gptsniffer_code_settingB.csv" || \
       ! -f "$PREDS_DIR/droiddetect_zeroshot.csv" ]] && \
        echo "  Setting B prediction CSVs not all present — skipping." || true
fi

# ---------------------------------------------------------------------------
# Phase 7 — Plots
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 7: Plots ==="

# Plots always regenerate — they are fast and should reflect the latest data.
# skip_if_done is intentionally NOT applied here.
if [[ -f "$RESULTS_DIR/stats_settingA/stats_results.json" ]]; then
    echo "  Setting A plots ..."
    python stats/plot.py \
        --results    "$RESULTS_DIR/stats_settingA/stats_results.json" \
        --modernbert "$PREDS_DIR/modernbert_diffs.csv" \
        --gptsniffer "$PREDS_DIR/gptsniffer_code_settingA.csv" \
        --droiddetect "$PREDS_DIR/droiddetect_zeroshot.csv" \
        --out        "$RESULTS_DIR/stats_settingA/"
else
    echo "  Setting A stats_results.json not found — skipping plots."
fi

if [[ -f "$RESULTS_DIR/stats_settingB/stats_results.json" ]]; then
    echo "  Setting B plots ..."
    EXTRA_PLOT_B=""
    if [[ -f "$PREDS_DIR/droiddetect_finetuned.csv" ]]; then
        EXTRA_PLOT_B="--droiddetect-finetuned $PREDS_DIR/droiddetect_finetuned.csv"
    fi
    python stats/plot.py \
        --results    "$RESULTS_DIR/stats_settingB/stats_results.json" \
        --modernbert "$PREDS_DIR/modernbert_diffs.csv" \
        --gptsniffer "$PREDS_DIR/gptsniffer_code_settingB.csv" \
        --droiddetect "$PREDS_DIR/droiddetect_zeroshot.csv" \
        $EXTRA_PLOT_B \
        --out        "$RESULTS_DIR/stats_settingB/"
else
    echo "  Setting B stats_results.json not found — skipping plots."
fi

echo ""
echo "=== Done! ==="
echo "Predictions : $PREDS_DIR/"
echo "Results     : $RESULTS_DIR/"
echo "  stats_settingA/   — Setting A statistical comparison + plots"
echo "  stats_settingB/   — Setting B statistical comparison + plots"
echo "  *_hvai.txt        — HumanVsAICode OOD reports"
echo "  *_daniotti.txt    — Daniotti OOD reports"
