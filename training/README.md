# Training & Evaluation

Self-contained scripts for training and evaluating AI code detectors.

## Usage

Both scripts use three subcommands: `train`, `eval-test`, and `eval-daniotti`.

### Training

```bash
python gptsniffer.py train
python gptsniffer.py train --max-samples 5000 --epochs 2
python modernbert.py train --max-length 2048
```

### Evaluation

```bash
# Eval on HumanVsAICode test set (all defaults)
python gptsniffer.py eval-test

# Eval on Daniotti real-world parquet (all defaults)
python gptsniffer.py eval-daniotti
```

## Directory Structure

```
harness/
├── gptsniffer.py                          # GPTSniffer (CodeBERT) harness
├── modernbert.py                          # ModernBERT-large harness
├── requirements.txt
├── data/
│   ├── humanvsaicode_python/              # Training & test data. Unzip humanvsaicode_python.zip
│   │   └── CONF/
│   │       ├── training_data/             # 804,535 .py files
│   │       └── testing_data/              # 201,134 .py files
│   └── final_data_2/                      # Daniotti real-world eval data. Unzip final_data_2.zip
│       └── pyfunctions_ai_classified.parquet
├── gptsniffer_output/                     # Created by training / place pretrained weights here. Unzip gptsniffer_output.zip
│   └── final_model/                       # Model weights, config, tokenizer.
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
└── modernbert_output/                     # Same structure as above
    └── final_model/
```

### Data

- **HumanVsAICode**: Individual `.py` files named `{label}_{source}_{id}.py` (0=AI, 1=human). Python coding challenge solutions.
  - Source: [OSS-Forge/HumanVsAICode](https://huggingface.co/datasets/OSS-Forge/HumanVsAICode)
- **Daniotti parquet**: ~2,000 labeled real-world Python functions. Columns: `modified_blocks` (code), `true_label` ("human" or "ai").

### Model Weights

**Training from scratch**: weights are saved to `{gptsniffer,modernbert}_output/final_model/` automatically after training completes.

**Using pretrained weights**: place the model directory (containing `config.json`, `model.safetensors`, and tokenizer files) at the default path:
- GPTSniffer: `gptsniffer_output/final_model/`
- ModernBERT: `modernbert_output/final_model/`

Or pass `--model-dir <path>` to the eval commands.

## Scripts

### gptsniffer.py

GPTSniffer (CodeBERT-base) — the best detector in our evaluation.

| Setting | Default |
|---------|---------|
| Base model | `microsoft/codebert-base` |
| Max tokens | 512 |
| Batch size | 16 |
| Learning rate | 5e-5 |
| Warmup | 100 steps |
| Best metric | accuracy |

### modernbert.py

ModernBERT-large — longer context alternative.

| Setting | Default |
|---------|---------|
| Base model | `answerdotai/ModernBERT-large` |
| Max tokens | 1024 (up to 8192 via `--max-length`) |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Warmup | 10% of training |
| Best metric | f1 |
| Grad accumulation | `max(1, 16 // batch_size)` |
