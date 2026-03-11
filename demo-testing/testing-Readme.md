# Demo Testing — End-to-End Pipeline

This folder is a **self-contained demo** of the VibeCodeReporter pipeline. You can run it from scratch: collect diffs from repos listed in `repo-csvs/`, run AI vs Human detection, then explore results with `info.py`, `CDA.py`, and `hyp.py`.

---

## Folder layout

```
demo-testing/
├── repo-csvs/          # Input: CSV(s) listing GitHub repos per domain (url/repo, domain, etc.)
├── collect.py          # Step 1: Clone repos, extract pre/post-LLM diffs → output/
├── analyze_diffs.py    # Step 2: Run AI detector on diffs → results/ (predictions CSVs)
├── info.py             # Step 3: Pre vs post comparison report + 5 figures per domain
├── CDA.py              # Optional: Full confirmatory data analysis (all domains)
├── hyp.py              # Optional: Hypothesis testing (all domains)
├── testing-Readme.md   # This file
├── requirements.txt    # Python dependencies for this folder
├── output/             # Created by collect.py: <DOMAIN>-prellm/, <DOMAIN>-postllm/ (.diff files)
├── results/            # Created by analyze_diffs.py: *_predictions.csv; then by info/CDA/hyp
├── repos/              # Created by collect.py: bare git clones
└── model/              # You must add: tokenizer.json + model.safetensors (AI detection model)
```

---

## Prerequisites

1. **Python 3.10+** and **Git** on your PATH.
2. **Model files**: Place the AI detection model in `demo-testing/model/`:
   - `tokenizer.json`
   - `model.safetensors`  
   (Copy from the main repo `model/` folder if you have it.)
3. **Install dependencies** (from inside `demo-testing/`):

   ```bash
   pip install -r requirements.txt
   ```

---

## Step-by-step workflow

### 1. Collect diffs — `collect.py`

- **Input**: `repo-csvs/*.csv` (each CSV = one domain; must have a column `url` or `repo` with GitHub URLs).
- **Output**: `output/<DOMAIN>-prellm/` and `output/<DOMAIN>-postllm/` containing `.diff` files (one per commit).
- **Config**: At the top of `collect.py`, edit `CSV_FILES` to enable the domain(s) you want (e.g. `CSV_DIR / "FINANCE.csv"`).

**Run:**

```bash
python collect.py
```

- Commits before Dec 2022 → `*-prellm`; from Dec 2022 onward → `*-postllm`.
- Creates `repos/` (bare clones) and `output/<DOMAIN>-prellm|postllm/`.

---

### 2. Run AI detection — `analyze_diffs.py`

- **Input**: `output/<DOMAIN>-prellm/` and `output/<DOMAIN>-postllm/` (from step 1).
- **Output**: `results/<DOMAIN>-prellm_predictions.csv` and `results/<DOMAIN>-postllm_predictions.csv` (per-file AI/human confidence and prediction).
- **Config**: Set `DEMO_DOMAIN` at the top of `analyze_diffs.py` to match the domain you used in `collect.py` (e.g. `"FINANCE"`). Folder names must be `output/<DOMAIN>-prellm` and `output/<DOMAIN>-postllm`.

**Run:**

```bash
python analyze_diffs.py
```

- Requires `model/` with `tokenizer.json` and `model.safetensors`.
- Writes prediction CSVs and summary stats into `results/`.

---

### 3. Pre vs post comparison — `info.py`

- **Input**: `results/` (must contain paired `*-prellm_predictions.csv` and `*-postllm_predictions.csv`).
- **Output**: For each domain pair, creates `results/<DOMAIN>_summary_results/` with:
  - `report.txt`
  - `01_ai_rate_comparison.png`
  - `02_bubble_scatter.png`
  - `03_distribution_shift.png`
  - `04_top_repos_delta.png`
  - `05_confidence_buckets.png`

**Run:**

```bash
python info.py
```

- If you omit the path, it uses `results/` in this folder. To use another folder:  
  `python info.py path/to/results`

---

### 4. Full CDA (all domains) — `CDA.py`

- **Input**: `results/` with **one subfolder per domain**, each containing `*prellm_predictions.csv` and `*postllm_predictions.csv`.  
  Example: `results/FINANCE/FINANCE-prellm_predictions.csv` and `results/FINANCE/FINANCE-postllm_predictions.csv`.
- **Output**: `results/CDA_results/figures/` (F01–F10) and `results/CDA_results/full_statistical_report.txt`.

**Run:**

```bash
python CDA.py
```

- Default root is this folder’s `results/`. Optional: `python CDA.py path/to/results`.  
- If you only ran one domain, create e.g. `results/FINANCE/` and put the two prediction CSVs there so CDA sees them.

---

### 5. Hypothesis testing (all domains) — `hyp.py`

- **Input**: Same as CDA — `results/` with **one subfolder per domain** containing the pre/post prediction CSVs.
- **Output**: `results/hypothesis_results/figures/` (H1–H10 + summary) and `results/hypothesis_results/hypothesis_report.txt`.

**Run:**

```bash
python hyp.py
```

- Default root is this folder’s `results/`. Optional: `python hyp.py path/to/results`.

---

## File summary

| File            | Purpose |
|-----------------|--------|
| **repo-csvs/**  | Input CSVs: one per domain, with repo URLs. Used only by `collect.py`. |
| **collect.py**  | Clone repos from `repo-csvs`, split commits into pre/post-LLM, write `.diff` files into `output/`. |
| **analyze_diffs.py** | Run the AI detection model on `output/` diffs; write prediction CSVs to `results/`. |
| **info.py**     | Build pre vs post comparison (report + 5 figures) per domain from `results/`. |
| **CDA.py**      | Confirmatory data analysis across all domains: figures + full statistical report. |
| **hyp.py**      | Hypothesis tests (H1–H10) across domains; figures + hypothesis report. |

---

## Results layout

- **info.py** looks for `*-prellm_predictions.csv` and `*-postllm_predictions.csv` **anywhere under** `results/` (flat or in subfolders). The default `results/` from `analyze_diffs.py` (CSVs directly in `results/`) works.
- **CDA.py** and **hyp.py** expect **one subfolder per domain** under `results/`, with the two CSVs inside that subfolder.  
  Example: after `analyze_diffs.py` you have `results/FINANCE-prellm_predictions.csv` and `results/FINANCE-postllm_predictions.csv`. To use CDA/hyp, create `results/FINANCE/` and copy (or move) those two files into it.

---

## Quick test (after you have model and data)

1. Enable one CSV in `collect.py` (e.g. `FINANCE.csv`) and run `python collect.py`.
2. Set `DEMO_DOMAIN = "FINANCE"` in `analyze_diffs.py`, then run `python analyze_diffs.py`.
3. Run `python info.py` to get the comparison report and figures for that domain.
4. For CDA/hyp: create `results/FINANCE/` and put the two prediction CSVs there, then run `python CDA.py` and `python hyp.py`.

All paths are relative to the `demo-testing` folder, so you can move or clone it and run from anywhere.
