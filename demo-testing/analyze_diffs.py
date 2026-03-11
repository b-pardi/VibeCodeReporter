"""
Demo: Run AI vs Human detection on .diff files produced by collect.py.
Uses pre/post folders under output/; writes predictions to results/.
Set DEMO_DOMAIN below to match the domain(s) you enabled in collect.py (CSV stem).
"""
import os
import random
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, AutoConfig
from safetensors.torch import load_file
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

# Demo root = folder containing this script (demo-testing)
_REPO_ROOT = Path(__file__).resolve().parent
# Must match domain from collect.py CSV stem (e.g. FINANCE, CYBERSECURITY)
DEMO_DOMAIN = "FINANCE"
PRE_DIR    = _REPO_ROOT / "output" / f"{DEMO_DOMAIN}-prellm"
POST_DIR   = _REPO_ROOT / "output" / f"{DEMO_DOMAIN}-postllm"
MODEL_PATH = str(_REPO_ROOT / "model")
OUT_DIR    = _REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE       = 64
FILE_WORKERS     = 16
TOK_WORKERS      = 8
PREFETCH_BATCHES = 24

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHINDUCTOR_DISABLE"]  = "1"
os.environ["OMP_NUM_THREADS"]        = "8"

torch.set_num_threads(8)
torch.backends.cudnn.benchmark    = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
def make_tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(MODEL_PATH, "tokenizer.json"),
        pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", unk_token="[UNK]", model_max_length=1024,
    )
tokenizers = [make_tokenizer() for _ in range(TOK_WORKERS)]

print("Loading model...")
config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_labels        = 2
config.reference_compile = False

model = AutoModelForSequenceClassification.from_config(config)
weights = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
model.load_state_dict(weights, strict=False)
model = model.to(dtype=torch.float16, device=device)
model.eval()
print(f"Running on: {device}\n")

_file_pool = ThreadPoolExecutor(max_workers=FILE_WORKERS)

def read_file(f):
    try:
        return f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def tokenize_chunk(args):
    chunk, tok_idx = args
    tok   = tokenizers[tok_idx]
    texts = list(_file_pool.map(read_file, chunk))
    encoded = tok(texts, truncation=True, max_length=1024,
                  padding=True, return_tensors="pt")
    return {k: v.pin_memory() for k, v in encoded.items()}

def producer(files, queue: Queue):
    chunks = [
        (files[i : i + BATCH_SIZE], (i // BATCH_SIZE) % TOK_WORKERS)
        for i in range(0, len(files), BATCH_SIZE)
    ]
    with ThreadPoolExecutor(max_workers=TOK_WORKERS) as tok_pool:
        for inputs in tok_pool.map(tokenize_chunk, chunks):
            queue.put(inputs)
    queue.put(None)

def repo_from_filename(name: str) -> str:
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else name

def compute_stats(scores: list, label: str) -> dict:
    a = np.array(scores)
    return {
        "label":   label,
        "n":       len(a),
        "mean":    float(np.mean(a)),
        "median":  float(np.median(a)),
        "std":     float(np.std(a)),
        "min":     float(np.min(a)),
        "max":     float(np.max(a)),
        "p25":     float(np.percentile(a, 25)),
        "p75":     float(np.percentile(a, 75)),
        "p95":     float(np.percentile(a, 95)),
        "p99":     float(np.percentile(a, 99)),
        "high_conf_n":   int(np.sum(a > 0.9)),
        "high_conf_pct": float(np.mean(a > 0.9) * 100),
    }


def analyze_folder(folder):
    files = list(folder.glob("*.diff"))
    random.shuffle(files)
    total_files = len(files)
    file_names  = [f.name for f in files]

    print(f"{'='*40}")
    print(f"Folder : {folder.name}")
    print(f"Files  : {total_files:,}")
    print(f"{'='*40}\n")

    results = []
    queue = Queue(maxsize=PREFETCH_BATCHES)
    t = Thread(target=producer, args=(files, queue), daemon=True)
    t.start()

    batch_idx = 0
    with tqdm(total=total_files, desc=folder.name, unit="files") as pbar:
        while True:
            inputs = queue.get()
            if inputs is None:
                break

            batch_size_actual = inputs["input_ids"].shape[0]
            batch_files = file_names[batch_idx : batch_idx + batch_size_actual]
            batch_idx  += batch_size_actual

            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1)
            probs = probs.cpu().to(torch.float32).numpy()

            for fname, p in zip(batch_files, probs):
                ai_prob    = float(p[0])
                human_prob = float(p[1])
                label      = "AI" if ai_prob > human_prob else "Human"
                results.append((fname, ai_prob, human_prob, label))

            pbar.update(batch_size_actual)
            del inputs, logits, probs

    return results


def save_and_report(folder_name, results):
    csv_path = OUT_DIR / f"{folder_name}_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "repo", "ai_confidence", "human_confidence", "prediction"])
        for fname, ai_p, hum_p, label in results:
            writer.writerow([fname, repo_from_filename(fname), f"{ai_p:.6f}", f"{hum_p:.6f}", label])
    print(f"\n  Saved per-file CSV → {csv_path}")

    ai_scores    = [r[1] for r in results if r[3] == "AI"]
    human_scores = [r[2] for r in results if r[3] == "Human"]
    all_ai_probs = [r[1] for r in results]

    total = len(results)
    n_ai  = len(ai_scores)
    n_hum = len(human_scores)

    repo_stats = defaultdict(lambda: {"ai": 0, "human": 0, "ai_probs": [], "human_probs": []})
    for fname, ai_p, hum_p, label in results:
        repo = repo_from_filename(fname)
        repo_stats[repo]["ai_probs"].append(ai_p)
        repo_stats[repo]["human_probs"].append(hum_p)
        if label == "AI":
            repo_stats[repo]["ai"] += 1
        else:
            repo_stats[repo]["human"] += 1

    sep = "─" * 70
    print(f"\n{'='*70}")
    print(f"  RESULTS: {folder_name}")
    print(f"{'='*70}")
    print(f"\n  Total files      : {total:,}")
    print(f"  AI-generated     : {n_ai:,}  ({n_ai/total*100:.2f}%)")
    print(f"  Human-written    : {n_hum:,}  ({n_hum/total*100:.2f}%)")

    print(f"\n  {sep}")
    print(f"  CONFIDENCE SCORE DISTRIBUTIONS")
    print(f"  {sep}")
    for label, scores in [("AI predictions   (AI confidence)", ai_scores),
                           ("Human predictions (Human confidence)", human_scores),
                           ("All files         (AI probability)", all_ai_probs)]:
        if not scores:
            continue
        s = compute_stats(scores, label)
        print(f"\n  [{s['label']}]")
        print(f"    n         : {s['n']:,}")
        print(f"    mean      : {s['mean']:.4f}")
        print(f"    median    : {s['median']:.4f}")
        print(f"    std       : {s['std']:.4f}")
        print(f"    min/max   : {s['min']:.4f} / {s['max']:.4f}")
        print(f"    p25/p75   : {s['p25']:.4f} / {s['p75']:.4f}")
        print(f"    p95/p99   : {s['p95']:.4f} / {s['p99']:.4f}")
        print(f"    >90% conf : {s['high_conf_n']:,}  ({s['high_conf_pct']:.1f}%)")

    print(f"\n  {sep}")
    print(f"  CONFIDENCE BUCKETS  (AI probability distribution across all files)")
    print(f"  {sep}")
    buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    labels  = ["0.0–0.5 (Human)", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0 (High conf AI)"]
    probs   = np.array(all_ai_probs)
    for (lo, hi), lbl in zip(buckets, labels):
        n = int(np.sum((probs >= lo) & (probs < hi)))
        print(f"    {lbl:<28}  {n:>8,}  ({n/total*100:5.1f}%)")

    print(f"\n  {sep}")
    print(f"  PER-REPO BREAKDOWN")
    print(f"  {sep}")
    print(f"  {'REPO':<35}  {'TOTAL':>7}  {'AI':>7}  {'HUMAN':>7}  {'AI%':>7}  {'MEAN_AI_CONF':>13}")
    print(f"  {sep}")
    for repo, s in sorted(repo_stats.items(), key=lambda x: -(x[1]["ai"] + x[1]["human"])):
        t   = s["ai"] + s["human"]
        ai_pct = s["ai"] / t * 100 if t else 0
        mean_ai_conf = np.mean(s["ai_probs"]) if s["ai_probs"] else 0
        print(f"  {repo:<35}  {t:>7,}  {s['ai']:>7,}  {s['human']:>7,}  {ai_pct:>6.1f}%  {mean_ai_conf:>13.4f}")

    summary_path = OUT_DIR / f"{folder_name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis: {folder_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total: {total:,}\n")
        f.write(f"AI: {n_ai:,} ({n_ai/total*100:.2f}%)\n")
        f.write(f"Human: {n_hum:,} ({n_hum/total*100:.2f}%)\n\n")
        f.write("Per-repo:\n")
        for repo, s in sorted(repo_stats.items()):
            t = s["ai"] + s["human"]
            f.write(f"  {repo}: total={t}, ai={s['ai']}, human={s['human']}\n")
    print(f"\n  Saved summary     → {summary_path}")


for folder in [PRE_DIR, POST_DIR]:
    if not folder.exists():
        print(f"  [SKIP] Folder not found: {folder}")
        print(f"  Run collect.py first and set DEMO_DOMAIN = '{DEMO_DOMAIN}' to match your CSV.")
        continue
    results = analyze_folder(folder)
    save_and_report(folder.name, results)

print("\n\nAll done. Results saved to:", OUT_DIR)
print("Next: run info.py for pre/post comparison, or CDA.py / hyp.py for full analysis.")
