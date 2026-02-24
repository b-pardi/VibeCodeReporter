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
    print(classification_report(y_true, y_pred, target_names=target_names))


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
