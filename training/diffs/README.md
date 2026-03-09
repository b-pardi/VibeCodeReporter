# AI Commit Diff Collector

Collects labeled diffs (AI-assisted vs human-authored) from GitHub for
training a binary classifier. End-to-end pipeline from raw GitHub data to
train/test parquet files.

## Full Replication

```bash
# 1. Mine AI commits from GH Archive (Sept 2025, last month with data)
python collect.py mine --date 2025-09-01 --hours 720

# 2. Rank repos by AI commit density
python collect.py repos --input commits.jsonl --min-ai 3 --min-total 50

# 3. Clone AI repos + human baseline repos
python collect.py clone --input repos.jsonl --output /media/data/ghcommits/ --full --top 500
python collect.py clone --input human-baseline-repos.txt --output /media/data/ghcommits/ --full

# 4. Scan all repos (exclude temporal test repos)
python collect.py scan --repos /media/data/ghcommits/ --output /media/data/ghcommits-diffs/ --after 2018-01-01 --exclude test-repos-temporal.txt

# 5. Scan verified baseline repos for post-2023 human data (exclude temporal test repos)
python collect.py scan --repos /media/data/ghcommits/ --output /media/data/ghcommits-diffs/ \
  --after 2023-01-01 --max-commits 100000 \
  --include human-baseline-verified.txt --exclude test-repos-temporal.txt

# 6. Export to parquet (repo-level split, mixed-era human, capped)
python collect.py export --input /media/data/ghcommits-diffs/ --output /media/data/ghcommits-export/ \
  --repo-split --save-split test-repos.txt --max-human-per-repo 50000 --human-before 2099-12-31

# 7. Train
cd ../harness
python modernbert.py train --train-parquet /media/data/ghcommits-export/train.parquet \
  --output-dir ./modernbert_diffs_output

# 8. Temporal eval (post-2023 human from held-out repos)
python modernbert.py eval-diffs --diffs /media/data/ghcommits-temporal-eval/ \
  --model-dir modernbert_diffs_output/final_model
```

Steps 4 and 5 are additive — scanned repos are tracked in `scanned.txt` and
skipped on re-runs. Step 5 adds post-2023 human data from verified baseline
repos to address the temporal confound (see `baselines.md`).

