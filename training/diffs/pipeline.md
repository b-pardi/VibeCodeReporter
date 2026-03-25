# VibeCodeReporter: Full Pipeline


All collect steps run from `training/diffs/`. Training/eval steps run from `training/`.

---

## Phase 1 — Collect Data

### Step 1a: Mine AI-tagged commits from GH Archive

```bash
cd training/diffs
python collect.py mine \
    --date 2025-09-01 \
    --hours 720 \
    --output ~/data/git-data/commits.jsonl \
    --cache ~/data/git-data/gharchive-cache/ \
    --workers 4
```

Scans all of September 2025 — the last full month before GH Archive stopped including inline
commits in PushEvents (~2025-10-01). Produces `commits.jsonl`.

### Step 1b: Rank repos by AI commit density

```bash
python collect.py repos \
    --input ~/data/git-data/commits.jsonl \
    --output ~/data/git-data/repos.jsonl \
    --min-ai 3 \
    --min-total 50
```

### Step 1c: Clone repos (MUST use --full)

Clone with `--full` to download all blob objects upfront. Without it, git uses
`--filter=blob:none` (partial clones), causing `git diff` to fetch blobs on demand during scan —
making scan 100x+ slower.

```bash
# AI repos (top 500 by AI commit density)
python collect.py clone \
    --input ~/data/git-data/repos.jsonl \
    --output ~/data/git-data/ghcommits/ \
    --full \
    --top 500 \
    --skip skip.txt

# Human baseline repos (established pre-2023 projects)
python collect.py clone \
    --input human-baseline-repos.txt \
    --output ~/data/git-data/ghcommits/ \
    --full \
    --skip skip.txt
```

### Step 1d: Scan — Pass 1 (all repos, AI + pre-2023 human)

```bash
python collect.py scan \
    --repos ~/data/git-data/ghcommits/ \
    --output ~/data/git-data/ghcommits-diffs/ \
    --after 2018-01-01 \
    --max-commits 100000 \
    --exclude test-repos-temporal.txt \
    --workers 8
```

Excludes `test-repos-temporal.txt` to keep those repos as a held-out temporal OOD eval set.
Completed repos are tracked in `scanned.txt` and skipped on re-runs.

### Step 1e: Scan — Pass 2 (post-2023 verified human data)

```bash
python collect.py scan \
    --repos ~/data/git-data/ghcommits/ \
    --output ~/data/git-data/ghcommits-diffs/ \
    --after 2023-01-01 \
    --max-commits 100000 \
    --include human-baseline-verified.txt \
    --exclude test-repos-temporal.txt \
    --workers 8
```

Adds post-2023 human commits from repos verified to contain no AI-generated code
(`human-baseline-verified.txt`). This addresses the temporal confound: without this pass,
the human class would be entirely pre-2023, and a model could learn "old code = human"
rather than actual stylistic signals.

Steps 1d and 1e are additive — both write to the same output dir and share `scanned.txt`.

### (Optional) Check dataset stats

```bash
python collect.py stats --input ~/data/git-data/ghcommits-diffs/
```

---

## Phase 2 — Export to Parquet

### Step 2a: Export diff parquets (for ModernBERT)

```bash
python collect.py export \
    --input ~/data/git-data/ghcommits-diffs/ \
    --output ~/data/git-data/ghcommits-export/ \
    --human-before 2099-12-31 \
    --repo-split \
    --save-split test-repos-temporal.txt \
    --max-human-per-repo 50000
```

No date cutoff on human data (`--human-before 2099-12-31`) because Pass 2 already ensured
post-2023 human commits are clean. `--repo-split` ensures no repo appears in both train and
test. Produces `train.parquet` and `test.parquet`.

Label convention: **0 = AI-generated, 1 = human-written** throughout all parquets and harnesses.

### Step 2b: Export code-only parquets (for GPTSniffer + DroidDetect)

```bash
cd training
python diffs/collect.py code-export \
    --train-parquet ~/data/git-data/ghcommits-export/train.parquet \
    --test-parquet  ~/data/git-data/ghcommits-export/test.parquet \
    --train-out     ~/data/git-data/ghcommits-export/train_code.parquet \
    --test-out      ~/data/git-data/ghcommits-export/test_code.parquet
```

Strips diff markers (`+`/`-`/`@@`) from the text column, keeping only added and context lines.
Row order is preserved so diff and code parquets align for paired McNemar statistical tests.

---

## Phase 3 — Train Models

All commands from `training/`.

### Step 3a: Train ModernBERT on diffs

```bash
python modernbert.py train \
    --train-parquet ~/data/git-data/ghcommits-export/train.parquet \
    --output-dir ~/res/modernbert_diffs_output/ \
    --max-length 512 --epochs 1 --batch-size 16 \
    --gradient-checkpointing --bf16
```

### Step 3b (Setting A): Train GPTSniffer on HumanVsAICode (original paper setup)

```bash
python gptsniffer.py train \
    --data-dir ~/data/git-data/humanvsaicode_java \
    --output-dir ~/res/gptsniffer_hvai_output \
    --epochs 1 --bf16
```

No `--train-parquet` — uses the built-in HumanVsAICode dataset path. This is the zero-shot
baseline: tests how a model trained on the original benchmark generalizes to our pipeline data.

### Step 3c (Setting B): Train GPTSniffer on code parquet

```bash
python gptsniffer.py train \
    --train-parquet ~/data/git-data/ghcommits-export/train_code.parquet \
    --output-dir ~/res/gptsniffer_code_output \
    --max-length 512 --epochs 1 --batch-size 16 \
    --gradient-checkpointing --bf16
```

### Step 3d (Setting B): Fine-tune DroidDetect on code parquet

```bash
python droiddetect.py train \
    --train-parquet ~/data/git-data/ghcommits-export/train_code.parquet \
    --output-dir ~/res/droiddetect_output \
    --max-length 512 --epochs 1 --batch-size 16 \
    --gradient-checkpointing --bf16
```

---

## Phase 4 — In-Distribution Eval

```bash
mkdir -p ~/res/predictions

python modernbert.py eval-diffs \
    --model-dir ~/res/modernbert_diffs_output/final_model \
    --diffs     ~/data/git-data/ghcommits-export/test.parquet \
    --save-preds ~/res/predictions/modernbert_diffs.csv

python gptsniffer.py eval-diffs \
    --model-dir  ~/res/gptsniffer_hvai_output/final_model \
    --diffs      ~/data/git-data/ghcommits-export/test_code.parquet \
    --save-preds ~/res/predictions/gptsniffer_code_settingA.csv

python gptsniffer.py eval-diffs \
    --model-dir  ~/res/gptsniffer_code_output/final_model \
    --diffs      ~/data/git-data/ghcommits-export/test_code.parquet \
    --save-preds ~/res/predictions/gptsniffer_code_settingB.csv

python droiddetect.py eval-diffs \
    --diffs      ~/data/git-data/ghcommits-export/test_code.parquet \
    --save-preds ~/res/predictions/droiddetect_zeroshot.csv

python droiddetect.py eval-diffs \
    --model-dir  ~/res/droiddetect_output/final_model \
    --diffs      ~/data/git-data/ghcommits-export/test_code.parquet \
    --save-preds ~/res/predictions/droiddetect_finetuned.csv
```

---

## Phase 5 — OOD Eval

```bash
mkdir -p ~/res/results

# HumanVsAICode OOD
python modernbert.py eval-test \
    --model-dir ~/res/modernbert_diffs_output/final_model \
    --data-dir  ~/data/git-data/humanvsaicode_java \
    2>&1 | tee ~/res/results/modernbert_hvai.txt

python gptsniffer.py eval-test \
    --model-dir ~/res/gptsniffer_hvai_output/final_model \
    --data-dir  ~/data/git-data/humanvsaicode_java \
    2>&1 | tee ~/res/results/gptsniffer_a_hvai.txt

python gptsniffer.py eval-test \
    --model-dir ~/res/gptsniffer_code_output/final_model \
    --data-dir  ~/data/git-data/humanvsaicode_java \
    2>&1 | tee ~/res/results/gptsniffer_b_hvai.txt

python droiddetect.py eval-test \
    --data-dir ~/data/git-data/humanvsaicode_java \
    2>&1 | tee ~/res/results/droiddetect_zeroshot_hvai.txt

python droiddetect.py eval-test \
    --model-dir ~/res/droiddetect_output/final_model \
    --data-dir  ~/data/git-data/humanvsaicode_java \
    2>&1 | tee ~/res/results/droiddetect_finetuned_hvai.txt

# Daniotti OOD
python modernbert.py eval-daniotti \
    --model-dir ~/res/modernbert_diffs_output/final_model \
    2>&1 | tee ~/res/results/modernbert_daniotti.txt

python gptsniffer.py eval-daniotti \
    --model-dir ~/res/gptsniffer_hvai_output/final_model \
    2>&1 | tee ~/res/results/gptsniffer_a_daniotti.txt

python droiddetect.py eval-daniotti \
    2>&1 | tee ~/res/results/droiddetect_zeroshot_daniotti.txt
```

---

## Phase 6 — Statistical Analysis

```bash
# Setting A: zero-shot/original-training baselines vs our diff-trained model
python stats/analyze.py \
    --modernbert  ~/res/predictions/modernbert_diffs.csv \
    --gptsniffer  ~/res/predictions/gptsniffer_code_settingA.csv \
    --droiddetect ~/res/predictions/droiddetect_zeroshot.csv \
    --out         ~/res/results/stats_settingA/ \
    2>&1 | tee ~/res/results/stats_settingA.txt

# Setting B: all three trained on same data, only modality differs
python stats/analyze.py \
    --modernbert            ~/res/predictions/modernbert_diffs.csv \
    --gptsniffer            ~/res/predictions/gptsniffer_code_settingB.csv \
    --droiddetect           ~/res/predictions/droiddetect_zeroshot.csv \
    --droiddetect-finetuned ~/res/predictions/droiddetect_finetuned.csv \
    --out                   ~/res/results/stats_settingB/ \
    2>&1 | tee ~/res/results/stats_settingB.txt
```

---

## Phase 7 — Plots

Reads `stats_results.json` and the prediction CSVs to produce four PNG figures per setting.

```bash
# Setting A
python stats/plot.py \
    --results    ~/res/results/stats_settingA/stats_results.json \
    --modernbert ~/res/predictions/modernbert_diffs.csv \
    --gptsniffer ~/res/predictions/gptsniffer_code_settingA.csv \
    --droiddetect ~/res/predictions/droiddetect_zeroshot.csv \
    --out        ~/res/results/stats_settingA/

# Setting B (add fine-tuned DroidDetect if available)
python stats/plot.py \
    --results    ~/res/results/stats_settingB/stats_results.json \
    --modernbert ~/res/predictions/modernbert_diffs.csv \
    --gptsniffer ~/res/predictions/gptsniffer_code_settingB.csv \
    --droiddetect ~/res/predictions/droiddetect_zeroshot.csv \
    --droiddetect-finetuned ~/res/predictions/droiddetect_finetuned.csv \
    --out        ~/res/results/stats_settingB/
```

Output PNGs per setting directory:
- `fig_metrics_bar.png`        — Accuracy / F1 macro / F1(AI) / F1(human) with 95% CI bars
- `fig_confusion_matrices.png` — Per-model confusion matrices (row-normalised + raw counts)
- `fig_significance.png`       — Pairwise McNemar p-value heatmap with Cohen's h
- `fig_kappa.png`              — Inter-model prediction agreement (Cohen's κ)

---

## Or: Run Phases 4–7 All at Once

Once all models are trained, run from `training/`:

```bash
bash run_analysis.sh

# Smoke test with 1000 samples (fast sanity check)
bash run_analysis.sh --smoke-test

# Resume after interruption (skips steps whose output file already exists)
bash run_analysis.sh --resume
```

The script skips any model whose `final_model/` directory doesn't exist yet, so it's safe to run
partially if not all training is complete.
