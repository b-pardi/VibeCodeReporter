import os
import subprocess
import shutil
from pathlib import Path
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from collections import Counter
import pandas as pd


# Demo root = folder containing this script (demo-testing)
BASE_DIR   = Path(__file__).resolve().parent

CSV_DIR    = BASE_DIR / "repo-csvs"
CLONE_DIR  = BASE_DIR / "repos"
OUTPUT_DIR = BASE_DIR / "output"

REPO_WORKERS   = 12
COMMIT_WORKERS = 20

PRE_START  = datetime(2018, 1,   1,  tzinfo=timezone.utc)
PRE_END    = datetime(2022, 11, 30,  tzinfo=timezone.utc)

POST_START = datetime(2022, 12,  1,  tzinfo=timezone.utc)
POST_END   = datetime(2026,  3, 5,  tzinfo=timezone.utc)

MIN_ADDED_LINES = 8
MIN_DIFF_CHARS  = 200
MAX_DIFF_CHARS  = 4000

# Language → glob patterns
LANG_EXTENSIONS = {
    "Python":     ["*.py", "*.ipynb"],
    "C":          ["*.c"],
    "C++":        ["*.cpp"],
    "Java":       ["*.java"],
    "JavaScript": ["*.js", "*.jsx"],
    "TypeScript": ["*.ts", "*.tsx"],
}
ALL_PATTERNS = [pat for pats in LANG_EXTENSIONS.values() for pat in pats]

# Which domain CSVs to use. Uncomment one or more; domain name must match analyze_diffs DEMO_DOMAIN.
CSV_FILES = [
    CSV_DIR / "FINANCE.csv",
#    CSV_DIR / "CYBERSECURITY.csv",
#    CSV_DIR / "SCIENTIFIC_COMPUTING.csv",
]

os.environ["GIT_TERMINAL_PROMPT"] = "0"

CLONE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIFF_HEADER_RE = re.compile(
    r"^(diff --git .*|index [0-9a-f].*|--- .*|\+\+\+ .*)$",
    re.MULTILINE,
)

def strip_headers(text: str) -> str:
    return DIFF_HEADER_RE.sub("", text).strip()

# ================= ERA =================

def era_of(commit_date: datetime) -> str | None:
    if PRE_START  <= commit_date <= PRE_END:   return "pre"
    if POST_START <= commit_date <= POST_END:  return "post"
    return None


def load_repos_from_csvs() -> list[dict]:
    all_repos = []
    for csv_path in CSV_FILES:
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path}")
            continue

        domain = csv_path.stem
        df = pd.read_csv(csv_path)

        url_col = "url" if "url" in df.columns else "repo"
        if url_col not in df.columns:
            print(f"  [WARN] No url/repo column in {csv_path.name}")
            continue

        for _, row in df.iterrows():
            raw_url = str(row[url_col]).strip()
            if not raw_url.startswith("http"):
                raw_url = f"https://github.com/{raw_url}.git"
            if not raw_url.endswith(".git"):
                raw_url += ".git"
            repo_name = raw_url.rstrip("/").split("/")[-1].replace(".git", "")
            all_repos.append({
                "domain":    domain,
                "url":       raw_url,
                "repo_name": repo_name,
            })

        print(f"  Loaded {len(df):>4} repos from {csv_path.name}")
    return all_repos

# ================= GIT: CLONE =================

def is_valid_git_repo(path: Path) -> bool:
    """Check if the path is a healthy git repo (bare or normal)."""
    if not path.exists():
        return False
    r = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--git-dir"],
        capture_output=True, text=True,
    )
    return r.returncode == 0

def clone_repo(url: str, dest: Path):
    """
    Bare clone with blob filter.
    - Skips if dest is already a valid git repo.
    - Deletes and re-clones if dest exists but is broken.
    """
    if dest.exists():
        if is_valid_git_repo(dest):
            return   # already good
        else:
            print(f"  [WARN] Broken clone detected, re-cloning: {dest.name}")
            shutil.rmtree(dest, ignore_errors=True)

    subprocess.run(
        [
            "git", "clone",
            "--bare",
            url, str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3600,
    )


def get_commits(repo_path: Path) -> list[tuple[str, datetime]]:
    """
    Returns (sha, datetime_utc) for every non-merge commit.
    """
    r = subprocess.run(
        ["git", "-C", str(repo_path),
         "log", "--no-merges", "--format=%H|%at"],
        capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
        timeout=180,
    )

    if r.returncode != 0 or not r.stdout.strip():
        return []

    commits = []
    for line in r.stdout.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        sha, ts_str = line.split("|", 1)
        sha     = sha.strip()
        ts_str  = ts_str.strip()
        if not sha or not ts_str:
            continue
        try:
            commit_date = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
            commits.append((sha, commit_date))
        except (ValueError, OSError, OverflowError):
            continue

    return commits

# ================= GIT: DIFF =================

def has_parent(repo_path: Path, sha: str) -> bool:
    r = subprocess.run(
        ["git", "-C", str(repo_path),
         "rev-list", "--parents", "-n", "1", sha],
        capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
    )
    return len(r.stdout.strip().split()) > 1

def get_diff(repo_path: Path, sha: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo_path),
         "diff", f"{sha}^1", sha, "--"] + ALL_PATTERNS,
        capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
        timeout=60,
    )
    return r.stdout if r.returncode == 0 else ""

def count_added_lines(diff_text: str) -> int:
    return sum(
        1 for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

def process_commit(task: dict) -> tuple[str, bool]:
    repo_path  = task["repo_path"]
    sha        = task["sha"]
    era        = task["era"]
    out_path   = task["out_path"]

    if out_path.exists():
        return era, True

    try:
        if not has_parent(repo_path, sha):
            return era, False

        diff = get_diff(repo_path, sha)
        if not diff:
            return era, False

        if count_added_lines(diff) < MIN_ADDED_LINES:
            return era, False

        diff = strip_headers(diff)
        if not diff or len(diff) < MIN_DIFF_CHARS:
            return era, False

        if len(diff) > MAX_DIFF_CHARS:
            diff = diff[:MAX_DIFF_CHARS]

        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(diff)

        return era, True

    except Exception:
        return era, False


def clone_and_list(info: dict) -> dict:
    domain     = info["domain"]
    url        = info["url"]
    repo_name  = info["repo_name"]
    clone_path = CLONE_DIR / repo_name

    out_pre  = OUTPUT_DIR / f"{domain}-prellm"
    out_post = OUTPUT_DIR / f"{domain}-postllm"
    out_pre.mkdir(parents=True, exist_ok=True)
    out_post.mkdir(parents=True, exist_ok=True)

    try:
        clone_repo(url, clone_path)
    except Exception as e:
        print(f"  [ERR] Clone failed: {domain}/{repo_name} → {e}")
        return {"tasks": [], "domain": domain, "repo_name": repo_name}

    try:
        all_commits = get_commits(clone_path)
    except Exception as e:
        print(f"  [ERR] git log failed: {domain}/{repo_name} → {e}")
        return {"tasks": [], "domain": domain, "repo_name": repo_name}

    if not all_commits:
        print(f"  [WARN] No commits returned: {domain}/{repo_name} ")
        return {"tasks": [], "domain": domain, "repo_name": repo_name}

    pre_tasks  = []
    post_tasks = []

    for sha, commit_date in all_commits:
        era = era_of(commit_date)
        if era is None:
            continue
        out_folder = out_pre if era == "pre" else out_post
        task = {
            "repo_path": clone_path,
            "sha":       sha,
            "era":       era,
            "out_path":  out_folder / f"{repo_name}_{sha[:12]}.diff",
        }
        if era == "pre":
            pre_tasks.append(task)
        else:
            post_tasks.append(task)

    interleaved = []
    for p, q in zip_longest(pre_tasks, post_tasks):
        if p is not None: interleaved.append(p)
        if q is not None: interleaved.append(q)

    total_in_range = len(interleaved)
    total_commits  = len(all_commits)
    skipped        = total_commits - total_in_range

    print(
        f"  [READY] {domain}/{repo_name} — "
        f"pre: {len(pre_tasks):>5}  "
        f"post: {len(post_tasks):>5}  "
        f"total: {total_in_range:>6}  "
        f"(out-of-range skipped: {skipped})"
    )

    return {"tasks": interleaved, "domain": domain, "repo_name": repo_name}


def main():
    print("\n" + "="*70)
    print("  Multi-Domain Diff Extractor  —  Demo (demo-testing)")
    print("  Languages: Python · C · C++ · Java · JavaScript · TypeScript")
    print("="*70 + "\n")

    repos = load_repos_from_csvs()
    if not repos:
        print("No repos found. Check CSV_FILES and repo-csvs/.")
        return

    domain_counts = Counter(r["domain"] for r in repos)
    print(f"\n  {'Domain':<28} {'Repos':>6}")
    print(f"  {'-'*35}")
    for d, n in sorted(domain_counts.items()):
        print(f"  {d:<28} {n:>6}")
    print(f"  {'TOTAL':<28} {len(repos):>6}\n")

    print(f"Stage 1 — Cloning & listing commits ({REPO_WORKERS} workers)...\n")

    all_tasks = []
    with ThreadPoolExecutor(max_workers=REPO_WORKERS) as pool:
        futures = {pool.submit(clone_and_list, r): r for r in repos}
        for fut in as_completed(futures):
            result = fut.result()
            all_tasks.extend(result["tasks"])

    era_counts = Counter(t["era"] for t in all_tasks)
    print(f"\n  Total tasks queued : {len(all_tasks):>8}")
    print(f"  Pre-LLM  tasks     : {era_counts['pre']:>8}")
    print(f"  Post-LLM tasks     : {era_counts['post']:>8}\n")

    if not all_tasks:
        print("No commit tasks to process. Exiting.")
        return
    print(f"Stage 2 — Extracting diffs ({COMMIT_WORKERS} workers)...\n")

    saved_pre  = 0
    saved_post = 0
    total_done = 0
    report_every = max(500, len(all_tasks) // 40)

    with ThreadPoolExecutor(max_workers=COMMIT_WORKERS) as pool:
        futures = {pool.submit(process_commit, t): t for t in all_tasks}
        for fut in as_completed(futures):
            era, saved = fut.result()
            if saved:
                if era == "pre":  saved_pre  += 1
                else:             saved_post += 1
            total_done += 1
            if total_done % report_every == 0:
                print(
                    f"  [{total_done:>7}/{len(all_tasks)}]  "
                    f"pre saved: {saved_pre:>6}  "
                    f"post saved: {saved_post:>6}"
                )

    print("\n" + "="*70)
    print("  OUTPUT SUMMARY")
    print("="*70)
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        count = sum(1 for f in folder.iterdir() if f.suffix == ".diff")
        tag   = "[PRE ]" if "prellm" in folder.name else "[POST]"
        print(f"  {tag}  {folder.name:<40} {count:>8} .diff files")
    print("="*70)
    print(f"\n  Pre-LLM  diffs saved : {saved_pre}")
    print(f"  Post-LLM diffs saved : {saved_post}")
    print("\n✅  ALL DONE. Next: run analyze_diffs.py (requires model in ./model)\n")


if __name__ == "__main__":
    main()
