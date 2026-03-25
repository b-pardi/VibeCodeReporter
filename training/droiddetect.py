#!/usr/bin/env python3
"""
DroidDetect training and evaluation harness.

DroidDetect (Orel et al., EMNLP 2025) is a ModernBERT-based classifier
trained on DroidCollection (1M+ code samples, 7 languages, 43 LLMs).

HuggingFace: project-droid/DroidDetect-Large-Binary
Paper: https://arxiv.org/abs/2507.10583

LABEL CONVENTION WARNING:
  DroidDetect-Large-Binary uses 0=HUMAN, 1=MACHINE.
  This harness uses 0=AI, 1=human (HumanVsAICode convention).
  Predictions are always flipped:  harness_pred = 1 - raw_pred
  Training labels are also flipped: train_target = 1 - harness_label
  This preserves DroidDetect's original head orientation throughout.

Usage (Setting A — zero-shot):
    python droiddetect.py eval-diffs --diffs test_code.parquet \\
        --save-preds predictions/droiddetect_zeroshot.csv

Usage (Setting B — fine-tuned):
    python droiddetect.py train --train-parquet train_code.parquet \\
        --output-dir ./droiddetect_output
    python droiddetect.py eval-diffs --diffs test_code.parquet \\
        --model-dir droiddetect_output/final_model \\
        --save-preds predictions/droiddetect_finetuned.csv
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from common import (
    ParquetDataset, compute_metrics, print_eval_report,
    predict_batch, save_predictions,
    save_train_args, check_resume_args,
)


BASE_MODEL = 'project-droid/DroidDetect-Large-Binary'
BACKBONE    = 'answerdotai/ModernBERT-large'   # backbone used by DroidDetect-Large
HIDDEN_SIZE = 1024   # ModernBERT-large hidden dim
PROJ_DIM    = 256    # actual projection_dim from pytorch_model.bin weights (config.json is wrong)


class DroidModel(nn.Module):
    """Re-implementation of TLModel from project-droid/DroidDetect-Large-Binary.

    Architecture (matches pytorch_model.bin state dict keys exactly):
        text_encoder  : ModernBERT-large backbone
        text_projection : Linear(1024, 128) + ReLU
        classifier    : Linear(128, num_classes)

    Forward uses mean-pooling over last_hidden_state (same as original TLModel).
    Loss is cross-entropy only (original also adds 0.1× triplet loss; we skip
    that here to avoid the pytorch_metric_learning dependency).
    """

    def __init__(self, backbone, projection_dim=PROJ_DIM, num_classes=2):
        super().__init__()
        self.text_encoder    = backbone
        self.text_projection = nn.Linear(HIDDEN_SIZE, projection_dim)
        self.classifier      = nn.Linear(projection_dim, num_classes)
        self.num_classes     = num_classes

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over sequence length (matches original TLModel)
        pooled    = outputs.last_hidden_state.mean(dim=1)
        projected = F.relu(self.text_projection(pooled))
        logits    = self.classifier(projected)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes),
                                         labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_droid_model(model_dir_or_id, device):
    """Load DroidModel from a local fine-tuned checkpoint or zero-shot from HuggingFace.

    Handles both pytorch_model.bin (older HF format) and model.safetensors.
    Uses strict=False so extra keys from the original triplet-loss head are ignored.
    """
    model_path = Path(model_dir_or_id)

    if model_path.exists():
        # Fine-tuned local checkpoint
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        weights_file = model_path / 'pytorch_model.bin'
        if not weights_file.exists():
            weights_file = model_path / 'model.safetensors'
    else:
        # Zero-shot: download weights from HuggingFace
        from huggingface_hub import hf_hub_download
        tokenizer    = AutoTokenizer.from_pretrained(model_dir_or_id)
        weights_file = Path(hf_hub_download(model_dir_or_id, 'pytorch_model.bin'))

    backbone = AutoModel.from_pretrained(BACKBONE)
    model    = DroidModel(backbone)

    if str(weights_file).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_file))
    else:
        state_dict = torch.load(str(weights_file), map_location='cpu', weights_only=True)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored = [k for k in unexpected if k.startswith('additional_loss')]
    real_missing = [k for k in missing if not k.startswith('additional_loss')]
    if real_missing:
        print(f"  Warning: {len(real_missing)} unexpected missing keys")
    if ignored:
        print(f"  Note: ignored {len(ignored)} triplet-loss keys (not needed for inference)")

    model.to(device)
    return model, tokenizer
LABEL_NAMES = ['AI-generated (0)', 'Human-written (1)']


class DroidDataset(Dataset):
    """ParquetDataset variant that flips labels to DroidDetect orientation.

    DroidDetect-Large-Binary was trained with 0=HUMAN, 1=MACHINE.
    Our harness uses 0=AI, 1=human. This dataset flips labels so that
    fine-tuning continues in DroidDetect's original orientation, which
    keeps the pre-trained head weights aligned with their intended classes.
    Eval-time flip (pred = 1 - raw_pred) then maps back to harness convention.
    """

    def __init__(self, parquet_path, tokenizer, max_length=1024,
                 max_samples=None, seed=42, tokenize_batch=4096):
        import pandas as pd
        from tqdm import tqdm
        df = pd.read_parquet(parquet_path)
        if max_samples and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        texts = df['text'].tolist()
        # Flip: harness 0 (AI) -> DroidDetect 1 (MACHINE), 1 (human) -> 0 (HUMAN)
        self.labels = [1 - int(lbl) for lbl in df['label'].tolist()]

        n_machine = self.labels.count(1)
        n_human = self.labels.count(0)
        print(f"DroidDataset: {len(texts)} samples "
              f"[MACHINE(AI)={n_machine}, HUMAN={n_human}] from {parquet_path}")

        n = len(texts)
        self.input_ids = np.zeros((n, max_length), dtype=np.int32)
        self.attention_masks = np.zeros((n, max_length), dtype=np.int32)

        for i in tqdm(range(0, n, tokenize_batch),
                      desc="Tokenizing", unit="batch"):
            batch = texts[i: i + tokenize_batch]
            enc = tokenizer(batch, padding='max_length', max_length=max_length,
                            truncation=True)
            end = min(i + tokenize_batch, n)
            self.input_ids[i:end] = enc['input_ids']
            self.attention_masks[i:end] = enc['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'input_ids':      torch.tensor(self.input_ids[index], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[index], dtype=torch.long),
            'labels':         torch.tensor(self.labels[index], dtype=torch.long),
        }


def _flip_preds(preds):
    """Flip DroidDetect predictions to harness convention (0=AI, 1=human)."""
    return [1 - p for p in preds]


def cmd_train(args):
    """Fine-tune DroidDetect-Large-Binary on code-only parquet (Setting B)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = device.type == 'cuda'
    print(f"Using device: {device}")

    model_dir = Path(args.output_dir) / 'final_model'

    print(f"Loading {BASE_MODEL} (zero-shot weights → fine-tune) ...")
    model, tokenizer = load_droid_model(BASE_MODEL, device)

    if args.gradient_checkpointing:
        # DroidModel is a plain nn.Module; enable checkpointing on the backbone directly
        model.text_encoder.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled on backbone.")

    print(f"\nMax token length: {args.max_length}")
    print("\nLoading training data (parquet, labels flipped to DroidDetect orientation)...")
    train_dataset = DroidDataset(
        args.train_parquet, tokenizer, max_length=args.max_length,
        max_samples=args.max_samples, seed=args.seed)

    test_parquet = args.test_parquet
    if not test_parquet:
        test_parquet = str(Path(args.train_parquet).parent / 'test_code.parquet')

    test_max = None
    if args.max_samples:
        test_max = max(1000, args.max_samples // 4)

    print("\nLoading validation data (parquet)...")
    val_dataset = DroidDataset(
        test_parquet, tokenizer, max_length=args.max_length,
        max_samples=test_max, seed=args.seed)

    grad_accum = max(1, 16 // args.batch_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='steps',
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        optim='adamw_torch',
        learning_rate=args.learning_rate,
        save_total_limit=3,
        report_to='none',
        bf16=args.bf16 and cuda,
        fp16=(not args.bf16) and cuda,
        torch_compile=args.torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    if args.resume:
        check_resume_args(args.output_dir, args)
    else:
        save_train_args(args.output_dir, args)

    print(f"\n{'='*50}")
    print(f"Starting DroidDetect fine-tuning:")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Max tokens: {args.max_length}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * grad_accum})")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  NOTE: Labels are flipped to DroidDetect orientation (0=HUMAN,1=MACHINE).")
    print(f"        Predictions will be flipped back at eval time.")
    print(f"{'='*50}\n")

    try:
        trainer.train(resume_from_checkpoint=args.resume or None)
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Latest checkpoint in: {args.output_dir}")
        print(f"Resume with the same command plus --resume")
        return

    # Final validation (in DroidDetect orientation, then flip for report)
    print("\n" + "="*50)
    print("Final Validation")
    print("="*50)
    results = trainer.evaluate()
    print(f"Eval results (DroidDetect orientation): {results}")

    predictions = trainer.predict(val_dataset)
    raw_pred = np.argmax(predictions.predictions, axis=-1)
    raw_true = predictions.label_ids
    # Flip both back to harness convention for the report
    y_pred = 1 - raw_pred
    y_true = 1 - raw_true
    print_eval_report(y_true, y_pred, target_names=LABEL_NAMES)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / 'pytorch_model.bin')
    tokenizer.save_pretrained(str(model_dir))
    print(f"\nModel saved to {model_dir}")


def cmd_eval_diffs(args):
    """Evaluate on a diffs directory or parquet (code-only for DroidDetect).

    Loads model and tokenizer. For zero-shot use, the model is loaded directly
    from HuggingFace (BASE_MODEL). For fine-tuned eval, pass --model-dir.
    Predictions are flipped from DroidDetect convention to harness convention.
    """
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model, tokenizer = load_droid_model(args.model_dir, device)

    path = Path(args.diffs)
    shas = None
    dates = None

    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
        codes  = df['text'].tolist()
        labels = df['label'].tolist()   # already in harness convention (0=AI, 1=human)
        repos  = df['repo'].tolist()  if 'repo'  in df.columns else None
        shas   = df['sha'].tolist()   if 'sha'   in df.columns else None
        dates  = df['date'].tolist()  if 'date'  in df.columns else None
    else:
        import json
        files  = sorted(path.glob("*.json"))
        codes  = []
        labels = []
        repos  = []
        for f in files:
            d = json.loads(f.read_text())
            diff = d.get("diff", "")
            if not diff:
                continue
            codes.append(diff)
            labels.append(0 if d.get("label", 0) == 1 else 1)
            repos.append(d.get("repo", "unknown"))

    n_human = labels.count(1)
    n_ai    = labels.count(0)
    print(f"Loaded {len(codes)} samples (human={n_human}, ai={n_ai})")

    if args.max_samples and args.max_samples < len(codes):
        rng_idx = random.sample(range(len(codes)), args.max_samples)
        codes  = [codes[i]  for i in rng_idx]
        labels = [labels[i] for i in rng_idx]
        if repos:  repos  = [repos[i]  for i in rng_idx]
        if shas:   shas   = [shas[i]   for i in rng_idx]
        if dates:  dates  = [dates[i]  for i in rng_idx]
        print(f"Subsampled to {len(codes)}")

    print(f"\n{'='*60}")
    print("DroidDetect Diffs Evaluation")
    print(f"{'='*60}")
    ml = getattr(args, 'max_length', 1024)
    bs = args.batch_size

    raw_preds = predict_batch(codes, model, tokenizer, device,
                              max_length=ml, batch_size=bs)
    # Flip DroidDetect orientation (0=HUMAN,1=MACHINE) to harness (0=AI,1=human)
    preds = _flip_preds(raw_preds)

    print_eval_report(labels, preds, target_names=LABEL_NAMES)

    save_preds_path = getattr(args, 'save_preds', None)
    if save_preds_path:
        save_predictions(labels, preds, save_preds_path,
                         sha=shas, repo=repos, date=dates)


def cmd_eval_test_droid(args):
    """Evaluate on HumanVsAICode test split using DroidModel."""
    import pandas as pd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model, tokenizer = load_droid_model(args.model_dir, device)

    test_path = Path(args.data_dir) / 'CONF' / 'testing_data'
    files = [f for f in test_path.iterdir() if f.is_file() and not f.name.startswith('.')]
    if hasattr(args, 'max_samples') and args.max_samples and args.max_samples < len(files):
        import random as _random
        _random.seed(args.seed)
        files = _random.sample(files, args.max_samples)
    codes, labels = [], []
    for fp in files:
        codes.append(fp.read_text(encoding='utf-8', errors='ignore'))
        labels.append(int(fp.name.split('_')[0]))
    print(f"HumanVsAICode test: {len(codes)} samples")

    if not codes:
        print(f"  No files found in {test_path} — skipping eval.")
        return

    print(f"\n{'='*60}")
    print("HumanVsAICode Test Evaluation (DroidDetect)")
    print(f"{'='*60}")
    raw_preds = predict_batch(codes, model, tokenizer, device,
                              max_length=args.max_length, batch_size=args.batch_size)
    preds = _flip_preds(raw_preds)
    print_eval_report(labels, preds, target_names=LABEL_NAMES)


def cmd_eval_daniotti_droid(args):
    """Evaluate on Daniotti real-world parquet using DroidModel."""
    import pandas as pd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model, tokenizer = load_droid_model(args.model_dir, device)

    df = pd.read_parquet(args.parquet)
    codes = df['modified_blocks'].tolist()
    labels = [1 if l == 'human' else 0 for l in df['true_label']]
    n_human = labels.count(1)
    n_ai = labels.count(0)
    print(f"Parquet: {len(codes)} samples (human={n_human}, ai={n_ai})")

    print(f"\n{'='*60}")
    print("Daniotti (Real-World) Evaluation (DroidDetect)")
    print(f"{'='*60}")
    raw_preds = predict_batch(codes, model, tokenizer, device,
                              max_length=args.max_length, batch_size=args.batch_size)
    preds = _flip_preds(raw_preds)
    print_eval_report(labels, preds, target_names=LABEL_NAMES)


def main():
    parser = argparse.ArgumentParser(
        description='DroidDetect (ModernBERT-Large-Binary) evaluation and fine-tuning harness')
    sub = parser.add_subparsers(dest='command', required=True)

    # -- train (Setting B: fine-tune on code-only parquet) --
    p_train = sub.add_parser('train', help='Fine-tune DroidDetect on code-only parquet')
    p_train.add_argument('--train-parquet', required=True,
                         help='train_code.parquet from collect.py code-export')
    p_train.add_argument('--test-parquet', default=None,
                         help='test_code.parquet (default: auto-detected next to train)')
    p_train.add_argument('--output-dir', default='./droiddetect_output',
                         help='Output directory for model and logs')
    p_train.add_argument('--max-samples', type=int, default=None,
                         help='Max training samples (default: all)')
    p_train.add_argument('--epochs', type=int, default=3,
                         help='Number of training epochs')
    p_train.add_argument('--batch-size', type=int, default=8, help='Batch size')
    p_train.add_argument('--max-length', type=int, default=1024,
                         help='Max token length (DroidDetect supports up to 8192)')
    p_train.add_argument('--learning-rate', type=float, default=2e-5,
                         help='Learning rate')
    p_train.add_argument('--gradient-checkpointing', action='store_true',
                         help='Enable gradient checkpointing to save VRAM at cost of ~20%% slower training')
    p_train.add_argument('--bf16', action='store_true',
                         help='Use bf16 instead of fp16 (recommended on Ampere+ GPUs, e.g. 3080 Ti)')
    p_train.add_argument('--torch-compile', action='store_true',
                         help='Enable torch.compile for kernel fusion (15-30%% speedup, slower first few steps)')
    p_train.add_argument('--save-steps', type=int, default=500,
                         help='Save a checkpoint every N steps (default: 500)')
    p_train.add_argument('--resume', action='store_true',
                         help='Resume from the latest checkpoint in --output-dir')
    p_train.add_argument('--seed', type=int, default=42, help='Random seed')

    # -- eval-diffs (both zero-shot and fine-tuned) --
    p_edf = sub.add_parser('eval-diffs',
                            help='Evaluate on diffs directory or parquet')
    p_edf.add_argument(
        '--model-dir', default=BASE_MODEL,
        help=f'Path to fine-tuned model dir OR HuggingFace model ID for zero-shot '
             f'(default: {BASE_MODEL})')
    p_edf.add_argument('--diffs', required=True,
                       help='Diffs directory or parquet file (use code-only parquet)')
    p_edf.add_argument('--max-length', type=int, default=1024,
                       help='Max token length (up to 8192)')
    p_edf.add_argument('--max-samples', type=int, default=None,
                       help='Max samples (default: all)')
    p_edf.add_argument('--batch-size', type=int, default=8, help='Batch size')
    p_edf.add_argument('--save-preds', default=None,
                       help='Save predictions CSV to this path (for statistical comparison)')

    # -- eval-test (HumanVsAICode) --
    p_et = sub.add_parser('eval-test',
                           help='Evaluate on HumanVsAICode test split')
    p_et.add_argument('--model-dir', default=BASE_MODEL,
                      help='Path to model dir or HuggingFace model ID')
    p_et.add_argument('--data-dir', default='data/humanvsaicode_java',
                      help='Path to dataset directory')
    p_et.add_argument('--max-length', type=int, default=1024,
                      help='Max token length')
    p_et.add_argument('--max-samples', type=int, default=None,
                      help='Max test samples (default: all)')
    p_et.add_argument('--batch-size', type=int, default=8, help='Batch size')
    p_et.add_argument('--seed', type=int, default=42, help='Random seed')

    # -- eval-daniotti --
    p_ed = sub.add_parser('eval-daniotti',
                           help='Evaluate on Daniotti real-world parquet')
    p_ed.add_argument('--model-dir', default=BASE_MODEL,
                      help='Path to model dir or HuggingFace model ID')
    p_ed.add_argument('--parquet',
                      default='data/final_data_2/pyfunctions_ai_classified.parquet',
                      help='Path to labeled parquet')
    p_ed.add_argument('--max-length', type=int, default=1024,
                      help='Max token length')
    p_ed.add_argument('--batch-size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval-diffs':
        cmd_eval_diffs(args)
    elif args.command == 'eval-test':
        cmd_eval_test_droid(args)
    elif args.command == 'eval-daniotti':
        cmd_eval_daniotti_droid(args)


if __name__ == '__main__':
    main()
