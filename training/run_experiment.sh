#!/usr/bin/env bash
# =============================================================================
# VibeCodeReporter: Unified Experiment Runner
# =============================================================================
# Runs the full comparison between ModernBERT (diffs), GPTSniffer (code),
# and DroidDetect (zero-shot + fine-tuned) across two experimental settings.
#
# Setting A: Original/zero-shot baselines vs our diff-trained model.
#            Tests real-world utility of existing models on our pipeline data.
# Setting B: All three trained on same data; only modality differs.
#            Isolates the diff-modality advantage.
#
# Prerequisites:
#   pip install -r requirements.txt
#   Data exported: EXPORT_DIR/train.parquet, EXPORT_DIR/test.parquet
#   Code-only export: EXPORT_DIR/train_code.parquet, EXPORT_DIR/test_code.parquet
#   HumanVsAICode data: data/humanvsaicode_python/CONF/{training_data,testing_data}/
#   Daniotti parquet: data/final_data_2/pyfunctions_ai_classified.parquet
#
# Usage:
#   bash run_experiment.sh                   # full run (hours/days on GPU)
#   bash run_experiment.sh --smoke-test      # smoke test with 1000 samples
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these paths
# ---------------------------------------------------------------------------
EXPORT_DIR="/media/data/ghcommits-export"
PREDS_DIR="predictions"
RESULTS_DIR="results"

MODERNBERT_DIFFS_OUT="./modernbert_diffs_output"
GPTSNIFFER_A_OUT="./gptsniffer_hvai_output"        # Setting A: trained on HumanVsAICode
GPTSNIFFER_B_OUT="./gptsniffer_code_output"        # Setting B: trained on code parquet
DROIDDETECT_OUT="./droiddetect_output"             # Setting B: fine-tuned

MAX_SAMPLES=""   # leave empty for full dataset
if [[ "${1:-}" == "--smoke-test" ]]; then
    MAX_SAMPLES="--max-samples 1000"
    echo "=== SMOKE TEST MODE (max 1000 samples) ==="
fi

mkdir -p "$PREDS_DIR" "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Export code-only parquets (if not already done)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: code-export ==="
if [[ ! -f "$EXPORT_DIR/train_code.parquet" ]]; then
    python diffs/collect.py code-export \
        --train-parquet "$EXPORT_DIR/train.parquet" \
        --test-parquet  "$EXPORT_DIR/test.parquet" \
        --train-out     "$EXPORT_DIR/train_code.parquet" \
        --test-out      "$EXPORT_DIR/test_code.parquet"
else
    echo "  code parquets already exist, skipping."
fi

# ---------------------------------------------------------------------------
# Step 2: Train ModernBERT on diffs (used in both settings)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Train ModernBERT on diffs ==="
python modernbert.py train \
    --train-parquet "$EXPORT_DIR/train.parquet" \
    --output-dir    "$MODERNBERT_DIFFS_OUT" \
    $MAX_SAMPLES

# ---------------------------------------------------------------------------
# Step 3: Setting A — Train GPTSniffer on HumanVsAICode (original paper setup)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3 (Setting A): Train GPTSniffer on HumanVsAICode ==="
python gptsniffer.py train \
    --output-dir "$GPTSNIFFER_A_OUT" \
    $MAX_SAMPLES

# ---------------------------------------------------------------------------
# Step 4: Setting B — Train GPTSniffer on code-only parquet
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4 (Setting B): Train GPTSniffer on code-only parquet ==="
python gptsniffer.py train \
    --train-parquet "$EXPORT_DIR/train_code.parquet" \
    --output-dir    "$GPTSNIFFER_B_OUT" \
    $MAX_SAMPLES

# ---------------------------------------------------------------------------
# Step 5: Setting B — Fine-tune DroidDetect on code-only parquet
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5 (Setting B): Fine-tune DroidDetect on code-only parquet ==="
python droiddetect.py train \
    --train-parquet "$EXPORT_DIR/train_code.parquet" \
    --output-dir    "$DROIDDETECT_OUT" \
    $MAX_SAMPLES

# ---------------------------------------------------------------------------
# Step 6: In-distribution eval — test parquet (generate prediction CSVs)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 6: In-distribution eval (test parquets) ==="

echo "  ModernBERT on test.parquet (diffs) ..."
python modernbert.py eval-diffs \
    --model-dir "$MODERNBERT_DIFFS_OUT/final_model" \
    --diffs     "$EXPORT_DIR/test.parquet" \
    --save-preds "$PREDS_DIR/modernbert_diffs.csv" \
    $MAX_SAMPLES

echo "  GPTSniffer (Setting B) on test_code.parquet ..."
python gptsniffer.py eval-diffs \
    --model-dir "$GPTSNIFFER_B_OUT/final_model" \
    --diffs     "$EXPORT_DIR/test_code.parquet" \
    --save-preds "$PREDS_DIR/gptsniffer_code_settingB.csv" \
    $MAX_SAMPLES

echo "  GPTSniffer (Setting A, HumanVsAICode-trained) on test_code.parquet ..."
python gptsniffer.py eval-diffs \
    --model-dir "$GPTSNIFFER_A_OUT/final_model" \
    --diffs     "$EXPORT_DIR/test_code.parquet" \
    --save-preds "$PREDS_DIR/gptsniffer_code_settingA.csv" \
    $MAX_SAMPLES

echo "  DroidDetect zero-shot on test_code.parquet ..."
python droiddetect.py eval-diffs \
    --diffs     "$EXPORT_DIR/test_code.parquet" \
    --save-preds "$PREDS_DIR/droiddetect_zeroshot.csv" \
    $MAX_SAMPLES

echo "  DroidDetect fine-tuned (Setting B) on test_code.parquet ..."
python droiddetect.py eval-diffs \
    --model-dir "$DROIDDETECT_OUT/final_model" \
    --diffs     "$EXPORT_DIR/test_code.parquet" \
    --save-preds "$PREDS_DIR/droiddetect_finetuned.csv" \
    $MAX_SAMPLES

# ---------------------------------------------------------------------------
# Step 7: OOD eval — HumanVsAICode (no prediction CSVs, report only)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 7: OOD eval — HumanVsAICode ==="

echo "  ModernBERT (diffs) ..."
python modernbert.py eval-test \
    --model-dir "$MODERNBERT_DIFFS_OUT/final_model" \
    $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/modernbert_hvai.txt"

echo "  GPTSniffer (Setting A, HumanVsAICode-trained) ..."
python gptsniffer.py eval-test \
    --model-dir "$GPTSNIFFER_A_OUT/final_model" \
    $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/gptsniffer_a_hvai.txt"

echo "  GPTSniffer (Setting B, code-parquet-trained) ..."
python gptsniffer.py eval-test \
    --model-dir "$GPTSNIFFER_B_OUT/final_model" \
    $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/gptsniffer_b_hvai.txt"

echo "  DroidDetect zero-shot ..."
python droiddetect.py eval-test \
    $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/droiddetect_zeroshot_hvai.txt"

echo "  DroidDetect fine-tuned ..."
python droiddetect.py eval-test \
    --model-dir "$DROIDDETECT_OUT/final_model" \
    $MAX_SAMPLES 2>&1 | tee "$RESULTS_DIR/droiddetect_finetuned_hvai.txt"

# ---------------------------------------------------------------------------
# Step 8: OOD eval — Daniotti parquet
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 8: OOD eval — Daniotti ==="

for MODEL_NAME MODEL_DIR in \
    "modernbert_diffs"     "$MODERNBERT_DIFFS_OUT/final_model" \
    "gptsniffer_settingA"  "$GPTSNIFFER_A_OUT/final_model" \
    "gptsniffer_settingB"  "$GPTSNIFFER_B_OUT/final_model"; do
    echo "  $MODEL_NAME ..."
done

python modernbert.py eval-daniotti \
    --model-dir "$MODERNBERT_DIFFS_OUT/final_model" 2>&1 | tee "$RESULTS_DIR/modernbert_daniotti.txt"

python gptsniffer.py eval-daniotti \
    --model-dir "$GPTSNIFFER_A_OUT/final_model" 2>&1 | tee "$RESULTS_DIR/gptsniffer_a_daniotti.txt"

python gptsniffer.py eval-daniotti \
    --model-dir "$GPTSNIFFER_B_OUT/final_model" 2>&1 | tee "$RESULTS_DIR/gptsniffer_b_daniotti.txt"

python droiddetect.py eval-daniotti 2>&1 | tee "$RESULTS_DIR/droiddetect_zeroshot_daniotti.txt"

python droiddetect.py eval-daniotti \
    --model-dir "$DROIDDETECT_OUT/final_model" 2>&1 | tee "$RESULTS_DIR/droiddetect_finetuned_daniotti.txt"

# ---------------------------------------------------------------------------
# Step 9: Statistical analysis — Setting A (zero-shot baselines)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 9: Statistical analysis — Setting A ==="
python stats/analyze.py \
    --modernbert  "$PREDS_DIR/modernbert_diffs.csv" \
    --gptsniffer  "$PREDS_DIR/gptsniffer_code_settingA.csv" \
    --droiddetect "$PREDS_DIR/droiddetect_zeroshot.csv" \
    --out         "$RESULTS_DIR/stats_settingA/" 2>&1 | tee "$RESULTS_DIR/stats_settingA.txt"

# ---------------------------------------------------------------------------
# Step 10: Statistical analysis — Setting B (same data, different modality)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 10: Statistical analysis — Setting B ==="
python stats/analyze.py \
    --modernbert           "$PREDS_DIR/modernbert_diffs.csv" \
    --gptsniffer           "$PREDS_DIR/gptsniffer_code_settingB.csv" \
    --droiddetect          "$PREDS_DIR/droiddetect_zeroshot.csv" \
    --droiddetect-finetuned "$PREDS_DIR/droiddetect_finetuned.csv" \
    --out                  "$RESULTS_DIR/stats_settingB/" 2>&1 | tee "$RESULTS_DIR/stats_settingB.txt"

echo ""
echo "=== All done! ==="
echo "Results in: $RESULTS_DIR/"
echo "  stats_settingA/stats_results.{md,json}  — Setting A statistical comparison"
echo "  stats_settingB/stats_results.{md,json}  — Setting B statistical comparison"
echo "  *_hvai.txt / *_daniotti.txt             — OOD eval reports"
