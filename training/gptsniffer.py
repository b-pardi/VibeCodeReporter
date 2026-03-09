#!/usr/bin/env python3
"""
GPTSniffer (CodeBERT) training and evaluation harness.

Usage:
    python gptsniffer.py train --epochs 3
    python gptsniffer.py eval-test
    python gptsniffer.py eval-daniotti
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from common import (
    CodeDataset, compute_metrics, print_eval_report,
    cmd_eval_test, cmd_eval_daniotti,
)


def cmd_train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = Path(args.data_dir)
    train_path = data_path / 'CONF' / 'training_data'
    test_path = data_path / 'CONF' / 'testing_data'
    model_dir = Path(args.output_dir) / 'final_model'

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    print("Loading CodeBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2)
    model.to(device)

    print("\nLoading training data...")
    train_dataset = CodeDataset(
        train_path, tokenizer, max_length=512,
        max_samples=args.max_samples, seed=args.seed)

    test_max = None
    if args.max_samples:
        test_max = max(1000, args.max_samples // 4)

    print("\nLoading validation data...")
    val_dataset = CodeDataset(
        test_path, tokenizer, max_length=512,
        max_samples=test_max, seed=args.seed)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        optim='adamw_torch',
        learning_rate=args.learning_rate,
        save_total_limit=2,
        report_to='none',
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n{'='*50}")
    print(f"Starting training:")
    print(f"  Model: microsoft/codebert-base")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"{'='*50}\n")

    trainer.train()

    # Final validation
    print("\n" + "="*50)
    print("Final Validation")
    print("="*50)
    results = trainer.evaluate()
    print(f"Eval results: {results}")

    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    print_eval_report(y_true, y_pred,
                      target_names=['AI-generated (0)', 'Human-written (1)'])

    # Save model
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"\nModel saved to {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='GPTSniffer (CodeBERT) training and evaluation harness')
    sub = parser.add_subparsers(dest='command', required=True)

    # -- train --
    p_train = sub.add_parser('train', help='Train a new model')
    p_train.add_argument('--data-dir', default='data/humanvsaicode_python',
                         help='Path to dataset directory (containing CONF/)')
    p_train.add_argument('--output-dir', default='./gptsniffer_output',
                         help='Output directory for model and logs')
    p_train.add_argument('--max-samples', type=int, default=None,
                         help='Max training samples (default: all)')
    p_train.add_argument('--epochs', type=int, default=3,
                         help='Number of training epochs')
    p_train.add_argument('--batch-size', type=int, default=16, help='Batch size')
    p_train.add_argument('--learning-rate', type=float, default=5e-5,
                         help='Learning rate')
    p_train.add_argument('--seed', type=int, default=42, help='Random seed')

    # -- eval-test --
    p_et = sub.add_parser('eval-test',
                          help='Evaluate on HumanVsAICode test split')
    p_et.add_argument('--model-dir', default='gptsniffer_output/final_model',
                      help='Path to saved model directory')
    p_et.add_argument('--data-dir', default='data/humanvsaicode_python',
                      help='Path to dataset directory')
    p_et.add_argument('--max-samples', type=int, default=None,
                      help='Max test samples (default: all)')
    p_et.add_argument('--batch-size', type=int, default=16, help='Batch size')
    p_et.add_argument('--seed', type=int, default=42, help='Random seed')

    # -- eval-daniotti --
    p_ed = sub.add_parser('eval-daniotti',
                          help='Evaluate on Daniotti real-world parquet')
    p_ed.add_argument('--model-dir', default='gptsniffer_output/final_model',
                      help='Path to saved model directory')
    p_ed.add_argument('--parquet',
                      default='data/final_data_2/pyfunctions_ai_classified.parquet',
                      help='Path to labeled parquet')
    p_ed.add_argument('--batch-size', type=int, default=16, help='Batch size')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval-test':
        cmd_eval_test(args, max_length=512, batch_size=16)
    elif args.command == 'eval-daniotti':
        cmd_eval_daniotti(args, max_length=512, batch_size=16)


if __name__ == '__main__':
    main()
