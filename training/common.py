"""
Shared utilities.

Labels follow HumanVsAICode convention: 0=AI, 1=human.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    """Lazy-loading dataset for HumanVsAICode .py files.

    Labels are raw from filenames: 0=AI, 1=human.
    """

    def __init__(self, directory, tokenizer, max_length=512,
                 max_samples=None, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_paths = []
        self.labels = []

        directory = Path(directory)
        all_files = list(directory.glob("*.py"))
        for fp in all_files:
            label = int(fp.name.split('_')[0])
            self.file_paths.append(fp)
            self.labels.append(label)

        print(f"Found {len(self.file_paths)} files in {directory}")

        # Huggingface Training shuffles data by default. No need to shuffle
        # all of the samples when max_samples is not specified.
        if max_samples and max_samples < len(self.file_paths):
            self._random_subset(max_samples, seed)

        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution: {label_counts}")

    def _random_subset(self, max_samples, seed):
        random.seed(seed)
        indices = list(range(len(self.file_paths)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        self.file_paths = [self.file_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        print(f"Subsampled to {len(self.file_paths)} samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        with open(self.file_paths[index], 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        inputs = self.tokenizer(
            code, padding='max_length', max_length=self.max_length,
            truncation=True, return_tensors=None,
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long),
        }


class ParquetDataset(Dataset):
    """Dataset backed by a parquet file with 'text' and 'label' columns.

    Labels follow HumanVsAICode convention: 0=AI, 1=human.
    """

    def __init__(self, parquet_path, tokenizer, max_length=512,
                 max_samples=None, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_parquet(parquet_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()

        print(f"Loaded {len(self.texts)} samples from {parquet_path}")

        if max_samples and max_samples < len(self.texts):
            random.seed(seed)
            indices = list(range(len(self.texts)))
            random.shuffle(indices)
            indices = indices[:max_samples]
            self.texts = [self.texts[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            print(f"Subsampled to {len(self.texts)} samples")

        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution: {label_counts}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.tokenizer(
            self.texts[index], padding='max_length', max_length=self.max_length,
            truncation=True, return_tensors=None,
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
    }


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def predict_batch(codes, model, tokenizer, device, max_length=512, batch_size=16):
    """Run batched inference. Returns raw model predictions (0=AI, 1=human)."""
    model.eval()
    all_preds = []
    for i in tqdm(range(0, len(codes), batch_size), desc='Predicting', unit='batch'):
        batch = codes[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
    return all_preds


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def print_eval_report(y_true, y_pred, target_names):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"  Accuracy: {acc:.4f}    F1: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names,
                                zero_division=0))


def load_model(model_dir, tokenizer_fallback=None):
    """Load a saved model and tokenizer.

    If tokenizer_fallback is set and the checkpoint lacks tokenizer files,
    falls back to loading from that model name (useful for ModernBERT checkpoints).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_dir = Path(model_dir)
    print(f"Loading model from {model_dir} ...")
    if tokenizer_fallback:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        except (OSError, ValueError):
            print(f"  Tokenizer not in checkpoint, falling back to {tokenizer_fallback}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_fallback)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    return model, tokenizer, device


def save_predictions(y_true, y_pred, out_path, sha=None, repo=None, date=None):
    """Save predictions CSV with columns: sha, repo, date, label, pred.

    Rows are sorted by sha for deterministic alignment across models in
    downstream statistical tests (McNemar requires paired samples).
    Pass None for sha/repo/date if metadata is unavailable (e.g. .py file eval).
    """
    n = len(y_true)
    data = {
        'sha':   sha   if sha   is not None else [''] * n,
        'repo':  repo  if repo  is not None else [''] * n,
        'date':  date  if date  is not None else [''] * n,
        'label': list(y_true),
        'pred':  list(y_pred),
    }
    df = pd.DataFrame(data)
    if sha is not None:
        df = df.sort_values('sha').reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


# ---------------------------------------------------------------------------
# Shared eval subcommands
# ---------------------------------------------------------------------------

LABEL_NAMES = ['AI-generated (0)', 'Human-written (1)']


def cmd_eval_test(args, max_length=512, batch_size=16, tokenizer_fallback=None):
    """Evaluate on HumanVsAICode test split. Both sides use raw labels."""
    model, tokenizer, device = load_model(args.model_dir, tokenizer_fallback)

    test_path = Path(args.data_dir) / 'CONF' / 'testing_data'
    files = list(test_path.glob('*.py'))
    if args.max_samples and args.max_samples < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.max_samples)
    codes, labels = [], []
    for fp in files:
        codes.append(fp.read_text(encoding='utf-8', errors='ignore'))
        labels.append(int(fp.name.split('_')[0]))
    print(f"HumanVsAICode test: {len(codes)} samples")

    print(f"\n{'='*60}")
    print("HumanVsAICode Test Evaluation")
    print(f"{'='*60}")
    ml = getattr(args, 'max_length', max_length)
    bs = args.batch_size if args.batch_size else batch_size
    preds = predict_batch(codes, model, tokenizer, device,
                          max_length=ml, batch_size=bs)
    print_eval_report(labels, preds, target_names=LABEL_NAMES)


def cmd_eval_diffs(args, max_length=512, batch_size=16, tokenizer_fallback=None):
    """Evaluate on a diffs directory or parquet from the collector pipeline.

    Reads JSON files from a directory or a parquet with 'text' and 'label' columns.
    Collector labels (1=AI, 0=human) are flipped to harness convention (0=AI, 1=human).
    """
    import json

    model, tokenizer, device = load_model(args.model_dir, tokenizer_fallback)

    per_repo = getattr(args, 'per_repo', False)

    path = Path(args.diffs)
    shas = None
    dates = None
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
        codes = df['text'].tolist()
        labels = df['label'].tolist()
        repos = df['repo'].tolist() if 'repo' in df.columns else None
        shas  = df['sha'].tolist()  if 'sha'  in df.columns else None
        dates = df['date'].tolist() if 'date' in df.columns else None
    else:
        # Read JSON files from directory
        files = sorted(path.glob("*.json"))
        codes = []
        labels = []
        repos = []
        for f in files:
            d = json.loads(f.read_text())
            diff = d.get("diff", "")
            if not diff:
                continue
            codes.append(diff)
            # Flip collector convention (1=AI,0=human) to harness (0=AI,1=human)
            labels.append(0 if d.get("label", 0) == 1 else 1)
            repos.append(d.get("repo", "unknown"))

    n_human = labels.count(1)
    n_ai = labels.count(0)
    print(f"Loaded {len(codes)} samples (human={n_human}, ai={n_ai})")

    if args.max_samples and args.max_samples < len(codes):
        import random
        random.seed(42)
        indices = random.sample(range(len(codes)), args.max_samples)
        codes = [codes[i] for i in indices]
        labels = [labels[i] for i in indices]
        if repos:
            repos = [repos[i] for i in indices]
        n_human = labels.count(1)
        n_ai = labels.count(0)
        print(f"Subsampled to {len(codes)} (human={n_human}, ai={n_ai})")

    print(f"\n{'='*60}")
    print("Diffs Evaluation")
    print(f"{'='*60}")
    ml = getattr(args, 'max_length', max_length)
    bs = args.batch_size if args.batch_size else batch_size
    preds = predict_batch(codes, model, tokenizer, device,
                          max_length=ml, batch_size=bs)
    print_eval_report(labels, preds, target_names=LABEL_NAMES)

    save_preds_path = getattr(args, 'save_preds', None)
    if save_preds_path:
        save_predictions(labels, preds, save_preds_path,
                         sha=shas, repo=repos, date=dates)

    if per_repo and repos:
        print(f"\n{'='*60}")
        print("Per-Repo Breakdown")
        print(f"{'='*60}")
        # Group by repo
        repo_data = {}
        for i, repo in enumerate(repos):
            if repo not in repo_data:
                repo_data[repo] = {'labels': [], 'preds': []}
            repo_data[repo]['labels'].append(labels[i])
            repo_data[repo]['preds'].append(preds[i])

        rows = []
        for repo in sorted(repo_data):
            rd = repo_data[repo]
            y_true = rd['labels']
            y_pred = rd['preds']
            n = len(y_true)
            n_h = y_true.count(1)
            n_a = y_true.count(0)
            acc = accuracy_score(y_true, y_pred)
            # Human FPR: human samples predicted as AI
            h_as_ai = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            h_fpr = h_as_ai / n_h if n_h else 0
            rows.append((repo, n, n_h, n_a, acc, h_fpr, h_as_ai))

        # Print table
        max_name = max(len(r[0]) for r in rows)
        hdr = f"{'Repo':<{max_name}}  {'N':>6}  {'Human':>6}  {'AI':>5}  {'Acc':>6}  {'H→AI':>6}  {'FPR':>6}"
        print(hdr)
        print("-" * len(hdr))
        for repo, n, n_h, n_a, acc, h_fpr, h_as_ai in rows:
            print(f"{repo:<{max_name}}  {n:>6}  {n_h:>6}  {n_a:>5}  {acc:>6.1%}  {h_as_ai:>6}  {h_fpr:>6.1%}")


def cmd_eval_daniotti(args, max_length=512, batch_size=16, tokenizer_fallback=None):
    """Evaluate on Daniotti real-world parquet.

    Labels mapped to model convention: 0=AI, 1=human. No inversion needed.
    """
    model, tokenizer, device = load_model(args.model_dir, tokenizer_fallback)

    df = pd.read_parquet(args.parquet)
    codes = df['modified_blocks'].tolist()
    # Match model convention: 0=AI, 1=human
    labels = [1 if l == 'human' else 0 for l in df['true_label']]
    n_human = labels.count(1)
    n_ai = labels.count(0)
    print(f"Parquet: {len(codes)} samples (human={n_human}, ai={n_ai})")

    print(f"\n{'='*60}")
    print("Daniotti (Real-World) Evaluation")
    print(f"{'='*60}")
    ml = getattr(args, 'max_length', max_length)
    bs = args.batch_size if args.batch_size else batch_size
    preds = predict_batch(codes, model, tokenizer, device,
                          max_length=ml, batch_size=bs)
    print_eval_report(labels, preds, target_names=LABEL_NAMES)
