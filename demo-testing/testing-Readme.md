# Demo Testing

Self-contained pipeline: collect diffs, run detection, then summarize pre vs post with `info.py`.

## Layout

```
demo-testing/
  repo-csvs/              Input CSVs per domain (url/repo)
  output/                 Diffs: <DOMAIN>-prellm/ and <DOMAIN>-postllm/ (.diff files)
  results/                Prediction CSVs and analysis output
  collect.py              Clone repos, extract pre/post diffs into output/
  analyze_diffs.py        Run detector on output/ diffs, write CSVs to results/<DOMAIN>/
  info.py                 Pre vs post report + figures per domain (reads results/)
  requirements.txt
  model/                  Optional: tokenizer.json + model.safetensors (for analyze_diffs.py)
```

## Quick start (uses existing results)

If `results/` already contains paired CSVs like `*-prellm_predictions.csv` and `*-postllm_predictions.csv`, you can run:

```bash
pip install -r requirements.txt
python info.py
```

Output: `results/<DOMAIN>_summary_results/` (report + figures).

## End-to-end (regenerate everything)

1. Configure domain CSVs in `collect.py` (`CSV_FILES`), then run:

```bash
python collect.py
```

2. Configure domain in `analyze_diffs.py` (`DEMO_DOMAIN`) and ensure `model/` exists, then run:

```bash
python analyze_diffs.py
```

This writes:
- `results/<DOMAIN>/<DOMAIN>-prellm_predictions.csv`
- `results/<DOMAIN>/<DOMAIN>-postllm_predictions.csv`

3. Summarize:

```bash
python info.py
```

## Requirements

Python 3.11+, Git. For `analyze_diffs.py`: `model/` with `tokenizer.json` and `model.safetensors`.
