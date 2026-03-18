import os
import json
import ast
import hashlib
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_CSV = os.path.expanduser("~/Desktop/combined_data.csv")
OUTPUT_JSONL = os.path.expanduser("~/Desktop/python_functions_all.jsonl")

LANGUAGE_COL = "language"
CODE_COL = "full_code"

META_COLS = [
    "repo_url", "repo_id", "commit_hash", "commit_date",
    "file_path", "change_type"
]

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base", use_fast=True)


def is_pre_2022(commit_date) -> bool:
    if not isinstance(commit_date, str) or len(commit_date) < 10:
        return False
    return commit_date[:10] < "2022-01-01"


def extract_top_level_functions(code: str):
    if not isinstance(code, str) or not code.strip():
        return []

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")

    try:
        tree = ast.parse(code)
    except Exception:
        return []

    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue

            start = node.lineno
            deco_linenos = []
            for d in getattr(node, "decorator_list", []) or []:
                ln = getattr(d, "lineno", None)
                if ln is not None:
                    deco_linenos.append(ln)
            if deco_linenos:
                start = min(deco_linenos)

            start_idx = max(start - 1, 0)
            end_idx = min(end, len(lines))
            text = "\n".join(lines[start_idx:end_idx]).strip("\n")

            if text.strip():
                funcs.append({
                    "function_name": getattr(node, "name", None),
                    "lineno": getattr(node, "lineno", None),
                    "end_lineno": end,
                    "text": text,
                })

    return funcs


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    chunksize = 5000

    total_rows = 0
    python_rows = 0
    functions_written = 0
    duplicates_skipped = 0
    seen_hashes = set()

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=chunksize), desc="CSV chunks"):
            total_rows += len(chunk)

            if LANGUAGE_COL not in chunk.columns:
                raise ValueError(f"CSV is missing required column: {LANGUAGE_COL}")
            if CODE_COL not in chunk.columns:
                raise ValueError(f"CSV is missing required column: {CODE_COL}")

            py = chunk[chunk[LANGUAGE_COL] == "Python"]
            python_rows += len(py)

            for _, row in py.iterrows():
                code = row.get(CODE_COL, None)
                funcs = extract_top_level_functions(code)
                if not funcs:
                    continue

                commit_date = row.get("commit_date", "")
                base = {
                    "language": "Python",
                    "pre_2022": is_pre_2022(str(commit_date)),
                    "tokenizer": "microsoft/graphcodebert-base",
                }

                for c in META_COLS:
                    base[c] = row.get(c, None)

                for fn in funcs:
                    text = fn["text"]
                    h = sha(text)

                    if h in seen_hashes:
                        duplicates_skipped += 1
                        continue
                    seen_hashes.add(h)

                    token_ids = tokenizer(
                        text,
                        add_special_tokens=True,
                        truncation=False
                    )["input_ids"]

                    out = {
                        **base,
                        "function_name": fn.get("function_name"),
                        "lineno": fn.get("lineno"),
                        "end_lineno": fn.get("end_lineno"),
                        "token_count": len(token_ids),
                        "text": text,
                    }

                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                    functions_written += 1

    print("Done.")
    print("Rows read:", total_rows)
    print("Python rows:", python_rows)
    print("Functions written:", functions_written)
    print("Duplicates skipped:", duplicates_skipped)
    print("Output:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()

