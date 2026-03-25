# VibeCodeReporter

## Overview

VibeCodeReporter is a research tool developed to analyze GitHub repositories by mining commit histories and diff files to detect and quantify AI-generated code contributions at the commit level. We use moderbert model to train on diff commits labeled by Claude and Human. 

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Conda](https://docs.conda.io/en/latest/) (recommended) or pip

### Installation

**Option 1 — Conda (recommended):**

```bash
git clone https://github.com/b-pardi/VibeCodeReporter.git
cd VibeCodeReporter
conda env create -f environment.yml
conda activate git-mining
```

**Option 2 — pip:**

```bash
git clone https://github.com/b-pardi/VibeCodeReporter.git
cd VibeCodeReporter
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `pydriller` | Git repository mining and commit diff extraction |
| `pandas` | Tabular data handling and CSV output |
| `tqdm` | Progress bars |
| `matplotlib` | Visualization |
| `lizard` | Code complexity analysis |
| `pygments` | Syntax highlighting / language detection |
| `sqlite` | (Planned) Database backend for commit data |

---

## Repository Structure

```
VibeCodeReporter/
├── main.py              # CLI entrypoint — ties all modules together
├── src/                 # Core pipeline modules (each runnable independently)
├── utils/               # Shared helper functions used across src scripts
├── consts/              # Dataclasses for configuration and frequently changing parameters
├── data/
│   └── repo_links.txt   # List of GitHub repo URLs to mine
├── output/              # CSV output from the mining pipeline
├── training/            # Training data and model artifacts (for AI detection)
├── demo-testing/        # Scripts and data for testing individual repos
├── requirements.txt
└── environment.yml
```

---

## How It Works

### Commit Mining

Given one or more GitHub URLs (directly or via `data/repo_links.txt`), VibeCodeReporter uses [PyDriller](https://pydriller.readthedocs.io/) to walk the commit history of each repository. For every commit it collects:

- Commit hash, author, timestamp, and message
- Per-file diff data (lines added/removed)
- ModernBert trained and tested

This data is written to CSV files in `output/` for downstream analysis.

### AI Detection

The detection model, ModernBert, will consume the mined diff files and apply ML-based classification to determine whether each code change is more consistent with human or AI-generated code. The goal is to produce per-author and per-repository breakdowns of estimated AI usage. 

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

Developed as a course project for **ECS 260 — Software Engineering** at UC Davis.
