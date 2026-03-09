#!/usr/bin/env python3
"""Collect AI-assisted and human-authored Python commit diffs from GitHub.

Pipeline:
  1. mine   — Scan GH Archive for commits with AI co-author/keyword markers.
  2. repos  — Rank repos by AI commit count and total activity.
  3. clone  — Shallow-clone top repos locally.
  4. scan   — Walk local git history, extract Python diffs with AI/human labels.

NOTE: GH Archive stopped including inline commits in PushEvents around Oct 2025.
Use --date before 2025-10-01 for the mine step (~400 AI commits/hr in mid-2025).

Usage:
    python collect.py mine --date 2025-06-15 --hours 168
    python collect.py repos --input commits.jsonl --min-ai 3 --min-total 50
    python collect.py clone --input repos.jsonl --output repos/
    python collect.py scan --repos repos/ --output diffs/
    python collect.py stats --input commits.jsonl
"""

import argparse
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# AI-marker detection
# ---------------------------------------------------------------------------

COAUTHOR_RE = [
    re.compile(r"co-authored-by:.*copilot", re.I),
    re.compile(r"co-authored-by:.*\bclaude\b", re.I),
    re.compile(r"co-authored-by:.*anthropic", re.I),
    re.compile(r"co-authored-by:.*\bcursor\b", re.I),
    re.compile(r"co-authored-by:.*\baider\b", re.I),
    re.compile(r"co-authored-by:.*\bdevin\b", re.I),
    re.compile(r"co-authored-by:.*openai", re.I),
    re.compile(r"co-authored-by:.*chatgpt", re.I),
    re.compile(r"co-authored-by:.*\bcody\b", re.I),
    re.compile(r"co-authored-by:.*\btabnine\b", re.I),
    re.compile(r"co-authored-by:.*gemini-code-assist", re.I),
    re.compile(r"co-authored-by:.*\bwindsurf\b", re.I),
    re.compile(r"co-authored-by:.*\bcodewhisperer\b", re.I),
    re.compile(r"co-authored-by:.*amazon.q\b", re.I),
]

KEYWORD_RE = [
    re.compile(
        r"\bgenerated\s+(?:by|with|using)\s+"
        r"(?:ai|copilot|claude|chatgpt|cursor|gpt|llm|cody|tabnine)",
        re.I,
    ),
    re.compile(
        r"\b(?:copilot|claude|chatgpt|cursor|aider|devin|cody|tabnine)"
        r"\s+generated\b",
        re.I,
    ),
    re.compile(r"\bai[- ](?:generated|assisted|written)\b", re.I),
]


def ai_match(message: str) -> dict | None:
    """Return match info if the commit message signals AI assistance."""
    for pat in COAUTHOR_RE:
        m = pat.search(message)
        if m:
            return {"type": "coauthor", "match": m.group().strip()}
    for pat in KEYWORD_RE:
        m = pat.search(message)
        if m:
            return {"type": "keyword", "match": m.group().strip()}
    return None


# ---------------------------------------------------------------------------
# GH Archive scanning
# ---------------------------------------------------------------------------

GHARCHIVE = "https://data.gharchive.org"


def fetch_hour(date_str: str, hour: int, cache_dir: Path) -> bytes | None:
    """Download one GH Archive hourly file, caching to disk."""
    fname = f"{date_str}-{hour}.json.gz"
    cached = cache_dir / fname

    if cached.exists():
        print(f"  {fname} (cached)", file=sys.stderr, end=" ", flush=True)
        return cached.read_bytes()

    url = f"{GHARCHIVE}/{fname}"
    print(f"  {url}", file=sys.stderr, end=" ", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-commit-collector/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except Exception as e:
        print(f"SKIP ({e})", file=sys.stderr)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def scan_hour(date_str: str, hour: int, cache_dir: Path, skip: set[str] | None = None):
    """Download one GH Archive hourly file. Yield (record, is_ai) tuples."""
    raw_gz = fetch_hour(date_str, hour, cache_dir)
    if raw_gz is None:
        return

    raw = gzip.decompress(raw_gz)

    n_ai = 0
    n_human = 0
    for line in raw.split(b"\n"):
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "PushEvent":
            continue

        repo = ev.get("repo", {}).get("name", "")
        if skip and repo in skip:
            continue
        ts = ev.get("created_at", "")
        commits = ev.get("payload", {}).get("commits", [])
        if not commits:
            continue

        for c in commits:
            sha = c.get("sha", "")
            msg = c.get("message") or ""
            if not sha:
                continue
            rec = {
                "sha": sha,
                "repo": repo,
                "message": msg,
                "author": c.get("author", {}).get("name", ""),
                "timestamp": ts,
            }
            m = ai_match(msg)
            if m:
                rec["label"] = 1
                rec["ai_match"] = m
                n_ai += 1
                yield rec, True
            else:
                rec["label"] = 0
                n_human += 1
                yield rec, False

    print(f"-> {n_ai} AI / {n_human} human", file=sys.stderr)


def cmd_mine(args):
    """Scan GH Archive hours, write all commits to JSONL."""
    start = datetime.strptime(args.date, "%Y-%m-%d")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    skip = load_skip(Path(args.skip)) if args.skip else set()
    if skip:
        print(f"Skipping {len(skip)} repos from {args.skip}", file=sys.stderr)

    total_ai = 0
    total_human = 0

    with open(out, "a") as f:
        for h in range(args.hours):
            dt = start + timedelta(hours=h)
            date_str = dt.strftime("%Y-%m-%d")
            hour = dt.hour

            ai_batch = []
            human_batch = []
            for rec, is_ai in scan_hour(date_str, hour, cache_dir, skip):
                if is_ai:
                    ai_batch.append(rec)
                else:
                    human_batch.append(rec)

            for rec in ai_batch:
                f.write(json.dumps(rec) + "\n")
            total_ai += len(ai_batch)

            # Sample human commits for negative examples
            rng = random.Random(42 + h)
            n_neg = max(len(ai_batch) * args.neg_ratio, 10)
            sampled = rng.sample(human_batch, min(n_neg, len(human_batch)))
            for rec in sampled:
                f.write(json.dumps(rec) + "\n")
            total_human += len(sampled)

    print(f"\nWrote {total_ai} AI + {total_human} human commits to {out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Repo ranking
# ---------------------------------------------------------------------------


def cmd_repos(args):
    """Aggregate commits.jsonl by repo, rank by AI activity + total size."""
    inp = Path(args.input)
    skip = load_skip(Path(args.skip)) if args.skip else set()
    repos: dict[str, dict] = {}

    with open(inp) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo = rec["repo"]
            if repo in skip:
                continue
            if repo not in repos:
                repos[repo] = {"repo": repo, "ai_commits": 0, "total_commits": 0,
                               "ai_shas": set(), "tools": set()}
            repos[repo]["total_commits"] += 1
            if rec.get("label") == 1:
                repos[repo]["ai_commits"] += 1
                repos[repo]["ai_shas"].add(rec["sha"])
                m = rec.get("ai_match", {}).get("match", "")
                for tool in ["copilot", "claude", "cursor", "aider", "chatgpt",
                             "devin", "cody", "tabnine"]:
                    if tool in m.lower():
                        repos[repo]["tools"].add(tool)

    # Filter
    filtered = []
    for r in repos.values():
        if r["ai_commits"] < args.min_ai:
            continue
        if r["total_commits"] < args.min_total:
            continue
        filtered.append(r)

    # Sort by AI commit count descending
    filtered.sort(key=lambda r: r["ai_commits"], reverse=True)

    if args.top:
        filtered = filtered[:args.top]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in filtered:
            row = {
                "repo": r["repo"],
                "ai_commits": r["ai_commits"],
                "total_commits": r["total_commits"],
                "ai_ratio": round(r["ai_commits"] / r["total_commits"], 3),
                "tools": sorted(r["tools"]),
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(filtered)} repos to {out}", file=sys.stderr)
    print(f"\nTop repos:", file=sys.stderr)
    for r in filtered[:20]:
        tools = ",".join(sorted(r["tools"])) or "?"
        print(f"  {r['ai_commits']:4d} AI / {r['total_commits']:5d} total  "
              f"[{tools}]  {r['repo']}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------


def load_skip(path: Path) -> set[str]:
    """Load skip.txt — one repo per line, # comments allowed."""
    if not path.exists():
        return set()
    skip = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            skip.add(line)
    return skip


def append_skip(path: Path, repo: str, reason: str):
    """Append a repo to skip.txt with a reason comment."""
    with open(path, "a") as f:
        f.write(f"{repo}  # {reason}\n")


def cmd_clone(args):
    """Bare-clone repos from repos.jsonl into a local directory."""
    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    skip_file = Path(args.skip)
    skip = load_skip(skip_file)

    # Accept JSONL (repos.jsonl) or plain text (one owner/repo per line)
    lines = [l.strip() for l in inp.read_text().splitlines() if l.strip()]
    if lines and lines[0].startswith("{"):
        repos = [json.loads(l) for l in lines]
    else:
        repos = [{"repo": l.split("#")[0].strip()}
                 for l in lines if l.split("#")[0].strip()]
    if args.top:
        repos = repos[:args.top]

    print(f"Cloning {len(repos)} repos into {out}/  "
          f"({len(skip)} in skip list)", file=sys.stderr)

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    cloned = skipped = errors = 0
    def cleanup_partial(dest: Path, out: Path):
        """Remove a partial clone dir, with safety checks."""
        if not dest.exists():
            return
        # Ensure dest is inside out and ends with .git
        try:
            dest.resolve().relative_to(out.resolve())
        except ValueError:
            return
        if not dest.name.endswith(".git") or len(dest.name) < 5:
            return
        shutil.rmtree(dest, ignore_errors=True)

    for r in repos:
        name = r["repo"]
        if "/" not in name or not name.strip():
            continue
        safe = name.replace("/", "_")
        dest = out / (safe + ".git")

        if name in skip:
            skipped += 1
            continue

        if dest.exists():
            print(f"  SKIP {name} (exists)", file=sys.stderr)
            skipped += 1
            continue

        url = f"https://github.com/{name}.git"
        print(f"  {name} ...", file=sys.stderr, end=" ", flush=True)
        try:
            cmd = ["git", "clone", "--bare", url, str(dest)]
            if not args.full:
                cmd.insert(3, "--filter=blob:none")
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=1800, check=True, env=env)
            print("OK", file=sys.stderr)
            cloned += 1
        except subprocess.CalledProcessError as e:
            msg = e.stderr.strip().split("\n")[-1][:80]
            print(f"FAIL ({msg})", file=sys.stderr)
            if "terminal prompts disabled" in e.stderr:
                append_skip(skip_file, name, "auth required")
            cleanup_partial(dest, out)
            errors += 1
        except subprocess.TimeoutExpired:
            print("TIMEOUT", file=sys.stderr)
            cleanup_partial(dest, out)
            errors += 1

    print(f"\nDone: {cloned} cloned, {skipped} skipped, {errors} errors", file=sys.stderr)


# ---------------------------------------------------------------------------
# Local scan
# ---------------------------------------------------------------------------

# Delimiter unlikely to appear in commit messages
LOG_SEP = "---COMMIT_BOUNDARY_f8a3b1e9---"

# ~1024 BERT tokens ≈ 4000 chars for code
MAX_DIFF_CHARS = 4000

# Diff header lines to strip (leak filenames)
DIFF_HEADER_RE = re.compile(
    r"^(diff --git .*|index [0-9a-f].*|--- .*|\+\+\+ .*)$", re.M
)


def strip_diff_headers(diff: str) -> str:
    """Remove diff/index/---/+++ lines, keeping only @@ hunks and code."""
    return DIFF_HEADER_RE.sub("", diff).strip()


def scan_repo(repo_dir: Path, seen_shas: set, max_commits: int = 0,
              after: str | None = None, before: str | None = None,
              exts: list[str] | None = None):
    """Walk git log of a local repo, yield dicts with diffs and labels.

    Deduplicates across branches via seen_shas (mutated in place).
    Works with both bare repos (foo.git/) and regular clones (foo/.git/).
    If exts is set (e.g. [".py", ".js"]), only diffs touching those extensions.
    Diffs are stripped of filenames and truncated to ~1024 BERT tokens.
    Streams git log via Popen to avoid loading entire history into memory.
    """
    repo_name = repo_dir.name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    # First, count commits for progress (lightweight, no body)
    count_cmd = ["git", "-C", str(repo_dir), "log", "--all", "--oneline"]
    if max_commits:
        count_cmd.append(f"-{max_commits}")
    if after:
        count_cmd.append(f"--after={after}")
    if before:
        count_cmd.append(f"--before={before}")
    try:
        count_result = subprocess.run(
            count_cmd, capture_output=True, text=True, timeout=120
        )
        n_entries = count_result.stdout.count("\n")
    except (subprocess.TimeoutExpired, Exception):
        n_entries = 0

    # %aI = author date ISO 8601
    fmt = f"%H%n%aI%n%s%n%b{LOG_SEP}"
    cmd = ["git", "-C", str(repo_dir), "log", "--all", f"--format={fmt}"]
    if max_commits:
        cmd.append(f"-{max_commits}")
    if after:
        cmd.append(f"--after={after}")
    if before:
        cmd.append(f"--before={before}")

    # Stream git log output to avoid loading entire history into memory
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    buf = []
    processed = 0
    try:
        for line in proc.stdout:
            if LOG_SEP in line:
                # Line may contain text before the separator
                before_sep = line.split(LOG_SEP)[0]
                if before_sep:
                    buf.append(before_sep)
                entry = "".join(buf).strip()
                buf = []
                if not entry:
                    continue
                lines = entry.split("\n", 3)
                if len(lines) < 3:
                    continue
                sha = lines[0].strip()
                date = lines[1].strip()
                subject = lines[2].strip()
                body = lines[3].strip() if len(lines) > 3 else ""
                message = f"{subject}\n{body}".strip()

                if not sha or len(sha) != 40:
                    continue

                processed += 1
                if processed % 500 == 0:
                    label_str = f"/{n_entries}" if n_entries else ""
                    print(f"    {repo_name}: {processed}{label_str} commits ...",
                          file=sys.stderr, flush=True)

                # Deduplicate across branches
                if sha in seen_shas:
                    continue
                seen_shas.add(sha)

                # Classify
                m = ai_match(message)
                label = 1 if m else 0

                # Get diff for this commit (optionally filtered by extension)
                diff_cmd = ["git", "-C", str(repo_dir), "diff",
                             f"{sha}~1", sha]
                if exts:
                    diff_cmd.append("--")
                    diff_cmd.extend(f"*{ext}" for ext in exts)
                try:
                    diff_result = subprocess.run(
                        diff_cmd, capture_output=True, text=True, timeout=30
                    )
                    diff_text = diff_result.stdout
                except (subprocess.TimeoutExpired, Exception):
                    diff_text = ""

                if not diff_text:
                    continue

                # Strip filenames and truncate to ~1024 BERT tokens
                diff_text = strip_diff_headers(diff_text)
                if not diff_text:
                    continue
                diff_text = diff_text[:MAX_DIFF_CHARS]

                rec = {
                    "sha": sha,
                    "repo": repo_name,
                    "date": date,
                    "message": message,
                    "label": label,
                    "diff": diff_text,
                }
                if m:
                    rec["ai_match"] = m
                yield rec
            else:
                buf.append(line)
    finally:
        proc.stdout.close()
        proc.wait()


def load_seen(path: Path) -> set[str]:
    """Load seen SHAs from file."""
    if not path.exists():
        return set()
    return set(line.strip() for line in path.read_text().splitlines() if line.strip())


def cmd_scan(args):
    """Scan cloned repos for commits, extract diffs with labels."""
    repos_dir = Path(args.repos)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    use_seen = args.seen is not None
    scanned_file = Path(args.scanned)

    # Load already-scanned repos
    scanned_repos = set()
    if scanned_file.exists():
        scanned_repos = set(
            line.strip() for line in scanned_file.read_text().splitlines()
            if line.strip()
        )

    # Global seen set — deduplicates across forks and re-runs
    if use_seen:
        seen_file = Path(args.seen)
        seen_shas = load_seen(seen_file)
        initial_seen = len(seen_shas)
        if initial_seen:
            print(f"Loaded {initial_seen} seen SHAs from {seen_file}", file=sys.stderr)
    else:
        seen_shas = set()
        initial_seen = 0

    # Detect both bare (foo.git/) and regular (foo/.git/) repos
    repo_dirs = sorted(
        d for d in repos_dir.iterdir()
        if d.is_dir() and ((d / "HEAD").exists() or (d / ".git").exists())
    )

    # Optional include/exclude filters (plain text, owner/repo or owner_repo)
    def load_repo_list(path):
        result = set()
        for line in Path(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                result.add(line.replace("/", "_"))
        return result

    include_set = load_repo_list(args.include) if args.include else None
    exclude_set = load_repo_list(args.exclude) if args.exclude else set()

    to_scan = [d for d in repo_dirs
               if d.name.removesuffix(".git") not in scanned_repos]
    if include_set:
        to_scan = [d for d in to_scan
                   if d.name.removesuffix(".git") in include_set]
    if exclude_set:
        to_scan = [d for d in to_scan
                   if d.name.removesuffix(".git") not in exclude_set]
    print(f"Scanning {len(to_scan)}/{len(repo_dirs)} repos in {repos_dir}/ "
          f"({len(scanned_repos)} already done)", file=sys.stderr)

    total_ai = 0
    total_human = 0
    sf = open(Path(args.seen), "a") if use_seen else None
    try:
        for repo_dir in to_scan:
            n_ai = n_human = 0

            repo_name = repo_dir.name
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]

            exts = args.ext if args.ext else None
            for rec in scan_repo(repo_dir, seen_shas, max_commits=args.max_commits,
                                 after=args.after, before=args.before, exts=exts):
                label = rec["label"]
                sha = rec["sha"]

                # Record SHA for cross-repo dedup
                if sf:
                    sf.write(sha + "\n")

                out_file = out / f"{label}_{repo_name}_{sha[:12]}.json"
                if not out_file.exists():
                    out_file.write_text(json.dumps(rec, indent=2))
                if label == 1:
                    n_ai += 1
                else:
                    n_human += 1

            total_ai += n_ai
            total_human += n_human
            print(f"  {repo_name}: {n_ai} AI + {n_human} human commits",
                  file=sys.stderr)

            # Mark repo as completed
            with open(scanned_file, "a") as cf:
                cf.write(repo_name + "\n")
    finally:
        if sf:
            sf.close()

    new_shas = len(seen_shas) - initial_seen
    print(f"\nTotal: {total_ai} AI + {total_human} human -> {out}/", file=sys.stderr)
    if use_seen:
        print(f"Seen SHAs: {initial_seen} loaded + {new_shas} new", file=sys.stderr)


# ---------------------------------------------------------------------------
# Language stats
# ---------------------------------------------------------------------------


def cmd_langs(args):
    """Count file extensions touched by AI vs human commits across cloned repos."""
    repos_dir = Path(args.repos)
    repo_dirs = sorted(
        d for d in repos_dir.iterdir()
        if d.is_dir() and ((d / "HEAD").exists() or (d / ".git").exists())
    )

    ai_exts: dict[str, int] = {}
    human_exts: dict[str, int] = {}
    total_ai = 0
    total_human = 0

    for repo_dir in repo_dirs:
        repo_name = repo_dir.name.removesuffix(".git")
        fmt = f"%H%n%s%n%b{LOG_SEP}"
        cmd = ["git", "-C", str(repo_dir), "log", "--all", f"--format={fmt}"]
        if args.after:
            cmd.append(f"--after={args.after}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            continue
        if result.returncode != 0:
            continue

        seen: set[str] = set()
        for entry in result.stdout.split(LOG_SEP):
            entry = entry.strip()
            if not entry:
                continue
            lines = entry.split("\n", 2)
            if len(lines) < 2:
                continue
            sha = lines[0].strip()
            if not sha or len(sha) != 40 or sha in seen:
                continue
            seen.add(sha)

            subject = lines[1].strip()
            body = lines[2].strip() if len(lines) > 2 else ""
            message = f"{subject}\n{body}".strip()
            is_ai = ai_match(message) is not None

            # Get changed file names (no blob fetch needed)
            ft_cmd = ["git", "-C", str(repo_dir), "diff-tree",
                       "--no-commit-id", "-r", "--name-only", sha]
            try:
                ft = subprocess.run(ft_cmd, capture_output=True, text=True, timeout=10)
                files = ft.stdout.strip().splitlines()
            except Exception:
                continue

            exts_seen = set()
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext:
                    exts_seen.add(ext)

            target = ai_exts if is_ai else human_exts
            for ext in exts_seen:
                target[ext] = target.get(ext, 0) + 1
            if is_ai:
                total_ai += 1
            else:
                total_human += 1

        print(f"  {repo_name}: {len(seen)} commits scanned", file=sys.stderr)

    print(f"\nAI commits: {total_ai}, Human commits: {total_human}\n")
    print(f"{'ext':>8}  {'AI':>6}  {'human':>6}  {'AI%':>5}")
    print(f"{'---':>8}  {'---':>6}  {'---':>6}  {'---':>5}")
    all_exts = sorted(set(ai_exts) | set(human_exts),
                       key=lambda e: ai_exts.get(e, 0), reverse=True)
    for ext in all_exts[:30]:
        a = ai_exts.get(ext, 0)
        h = human_exts.get(ext, 0)
        pct = a / (a + h) * 100 if (a + h) else 0
        print(f"{ext:>8}  {a:>6}  {h:>6}  {pct:>4.1f}%")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def cmd_stats(args):
    """Print stats for a JSONL file or a diffs directory."""
    path = Path(args.input)

    if path.is_dir():
        files = list(path.glob("*.json"))
        ai = [f for f in files if f.name.startswith("1_")]
        human = [f for f in files if f.name.startswith("0_")]
        total_lines = 0
        repos: dict[str, dict] = {}  # repo -> {min_date, max_date, ai, human}
        for f in files:
            d = json.loads(f.read_text())
            diff = d.get("diff", "")
            for fo in d.get("files", []):
                diff += fo.get("patch", "")
            total_lines += len(diff.splitlines())
            repo = d.get("repo", "unknown")
            date = d.get("date", "")[:10]  # YYYY-MM-DD
            label = d.get("label", 0)
            if repo not in repos:
                repos[repo] = {"min_date": date, "max_date": date,
                               "ai": 0, "human": 0, "pre2023": 0}
            r = repos[repo]
            if date and (not r["min_date"] or date < r["min_date"]):
                r["min_date"] = date
            if date and date > r["max_date"]:
                r["max_date"] = date
            if label == 1:
                r["ai"] += 1
            else:
                r["human"] += 1
            if date and date <= "2022-12-31":
                r["pre2023"] += 1

        pre_2023 = {name: r for name, r in repos.items()
                    if r["min_date"] <= "2022-12-31"}
        pre_2023_commits = sum(r["pre2023"] for r in pre_2023.values())

        print(f"Diff files:     {len(files)}")
        print(f"  AI (label=1): {len(ai)}")
        print(f"  Human (0):    {len(human)}")
        print(f"  Total diff lines: {total_lines}")
        print(f"Repos:          {len(repos)}")
        print(f"  With pre-2023 commits: {len(pre_2023)} ({pre_2023_commits} commits)")
        if pre_2023:
            print(f"  Pre-2023 repos:")
            for name, r in sorted(pre_2023.items(), key=lambda x: -x[1]["pre2023"]):
                print(f"    {name}: {r['pre2023']} pre-2023, "
                      f"{r['ai']} AI + {r['human']} human total, "
                      f"{r['min_date']} to {r['max_date']}")
        return

    ai_count = human_count = 0
    match_types: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") == 1:
                ai_count += 1
                mt = rec.get("ai_match", {}).get("type", "unknown")
                match_types[mt] = match_types.get(mt, 0) + 1
            else:
                human_count += 1

    print(f"AI commits:    {ai_count}")
    print(f"Human commits: {human_count}")
    print(f"Total:         {ai_count + human_count}")
    if match_types:
        print("\nAI match breakdown:")
        for k, v in sorted(match_types.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Export to parquet for training
# ---------------------------------------------------------------------------


def cmd_export(args):
    """Export diffs to train/test parquet files for classifier training.

    Selects AI-labeled diffs (any date) and human diffs from pre-2023 only.
    Balances classes, shuffles, and splits into train (90%) / test (10%).

    Output columns match harness convention:
      - text: diff content
      - label: 0=AI, 1=human  (HumanVsAICode convention)
      - repo, sha, date: metadata
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    diffs_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = args.human_before
    test_frac = args.test_split
    seed = args.seed

    ai_recs = []
    human_recs = []

    files = list(diffs_dir.glob("*.json"))
    print(f"Reading {len(files)} diff files ...", file=sys.stderr)

    for f in files:
        d = json.loads(f.read_text())
        date = d.get("date", "")[:10]
        label = d.get("label", 0)
        diff = d.get("diff", "")
        if not diff:
            continue

        rec = {
            "text": diff,
            "repo": d.get("repo", ""),
            "sha": d.get("sha", ""),
            "date": date,
        }

        if label == 1:
            # AI commit -> label 0 in harness convention
            rec["label"] = 0
            ai_recs.append(rec)
        elif date and date <= cutoff:
            # Pre-cutoff human commit -> label 1 in harness convention
            rec["label"] = 1
            human_recs.append(rec)

    print(f"AI diffs:          {len(ai_recs)}", file=sys.stderr)
    print(f"Human (pre-{cutoff[:4]}):  {len(human_recs)}", file=sys.stderr)

    rng = random.Random(seed)

    # Cap human commits per repo for diversity
    if args.max_human_per_repo:
        cap = args.max_human_per_repo
        by_repo: dict[str, list] = {}
        for r in human_recs:
            by_repo.setdefault(r["repo"], []).append(r)
        capped = []
        for repo, recs in by_repo.items():
            if len(recs) > cap:
                rng.shuffle(recs)
                recs = recs[:cap]
            capped.extend(recs)
        print(f"Human capped to {cap}/repo: {len(human_recs)} -> {len(capped)} "
              f"({len(by_repo)} repos)", file=sys.stderr)
        human_recs = capped

    def balance(a, h):
        minority = min(len(a), len(h))
        if len(a) > minority:
            rng.shuffle(a)
            a = a[:minority]
        if len(h) > minority:
            rng.shuffle(h)
            h = h[:minority]
        combined = a + h
        rng.shuffle(combined)
        return combined

    if args.repo_split:
        # Repo-level split: entire repos go to train or test, never both
        repo_ai: dict[str, list] = {}
        repo_human: dict[str, list] = {}
        for r in ai_recs:
            repo_ai.setdefault(r["repo"], []).append(r)
        for r in human_recs:
            repo_human.setdefault(r["repo"], []).append(r)
        all_repos = sorted(set(repo_ai) | set(repo_human))

        # Load or generate test repo set
        if args.load_split:
            split_path = Path(args.load_split)
            test_repo_set = set(
                l.strip() for l in split_path.read_text().splitlines() if l.strip())
            train_repo_set = set(all_repos) - test_repo_set
            print(f"Loaded test split: {len(test_repo_set)} test repos from {split_path}",
                  file=sys.stderr)
        else:
            rng.shuffle(all_repos)
            n_test_repos = max(1, int(len(all_repos) * test_frac))
            test_repo_set = set(all_repos[:n_test_repos])
            train_repo_set = set(all_repos[n_test_repos:])

        if args.save_split:
            split_path = Path(args.save_split)
            split_path.write_text("\n".join(sorted(test_repo_set)) + "\n")
            print(f"Saved test split to {split_path}", file=sys.stderr)

        print(f"Repo-level split: {len(all_repos)} repos, "
              f"{len(train_repo_set)} train, {len(test_repo_set)} test",
              file=sys.stderr)

        train_recs = balance(
            [r for repo in train_repo_set for r in repo_ai.get(repo, [])],
            [r for repo in train_repo_set for r in repo_human.get(repo, [])])
        test_recs = balance(
            [r for repo in test_repo_set for r in repo_ai.get(repo, [])],
            [r for repo in test_repo_set for r in repo_human.get(repo, [])])
    else:
        # Random split: commits from same repo may appear in both splits
        all_recs = balance(ai_recs, human_recs)
        n_test = max(1, int(len(all_recs) * test_frac))
        train_recs = all_recs[:-n_test]
        test_recs = all_recs[-n_test:]

    n_train = len(train_recs)
    n_test = len(test_recs)

    # Write parquet
    def write_parquet(recs, path):
        table = pa.table({
            "text": [r["text"] for r in recs],
            "label": [r["label"] for r in recs],
            "repo": [r["repo"] for r in recs],
            "sha": [r["sha"] for r in recs],
            "date": [r["date"] for r in recs],
        })
        pq.write_table(table, path)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    write_parquet(train_recs, train_path)
    write_parquet(test_recs, test_path)

    # Stats
    train_ai = sum(1 for r in train_recs if r["label"] == 0)
    train_human = n_train - train_ai
    test_ai = sum(1 for r in test_recs if r["label"] == 0)
    test_human = n_test - test_ai
    print(f"\nTrain: {n_train} ({train_ai} AI + {train_human} human) -> {train_path}",
          file=sys.stderr)
    print(f"Test:  {n_test} ({test_ai} AI + {test_human} human) -> {test_path}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Collect AI-assisted Python commit diffs from GitHub."
    )
    sub = p.add_subparsers(dest="cmd")

    # mine
    m = sub.add_parser("mine", help="Scan GH Archive for AI-tagged commits")
    m.add_argument("--date", required=True,
                    help="Start date YYYY-MM-DD (must be before 2025-10)")
    m.add_argument("--hours", type=int, default=24, help="Hours to scan")
    m.add_argument("--output", default="commits.jsonl")
    m.add_argument("--cache", default="/media/data/gharchive-cache/",
                    help="Dir to store raw .json.gz files")
    m.add_argument("--skip", default=None,
                    help="Skip list file (repos to ignore)")
    m.add_argument("--neg-ratio", type=int, default=3,
                    help="Human samples per AI commit (default 3)")
    m.set_defaults(func=cmd_mine)

    # repos
    r = sub.add_parser("repos", help="Rank repos by AI commit activity")
    r.add_argument("--input", required=True, help="JSONL from mine step")
    r.add_argument("--output", default="repos.jsonl")
    r.add_argument("--skip", default=None,
                    help="Skip list file (repos to ignore)")
    r.add_argument("--min-ai", type=int, default=3,
                    help="Min AI commits to include repo (default 3)")
    r.add_argument("--min-total", type=int, default=50,
                    help="Min total commits to include repo (default 50)")
    r.add_argument("--top", type=int, default=0, help="Keep only top N repos")
    r.set_defaults(func=cmd_repos)

    # clone
    c = sub.add_parser("clone", help="Clone repos locally")
    c.add_argument("--input", required=True, help="JSONL from repos step")
    c.add_argument("--output", default="repos/")
    c.add_argument("--skip", default="skip.txt",
                    help="Skip list file (default: skip.txt)")
    c.add_argument("--full", action="store_true",
                    help="Full clone (no blob filter). Larger but faster scans.")
    c.add_argument("--top", type=int, default=0, help="Clone only top N repos")
    c.set_defaults(func=cmd_clone)

    # scan
    sc = sub.add_parser("scan", help="Scan cloned repos for diffs")
    sc.add_argument("--repos", required=True, help="Directory of cloned repos")
    sc.add_argument("--output", default="diffs/")
    sc.add_argument("--seen", default=None,
                     help="Enable cross-repo SHA dedup with this file (e.g. seen-shas.txt)")
    sc.add_argument("--scanned", default="scanned.txt",
                     help="Track completed repos (default: scanned.txt)")
    sc.add_argument("--after", default=None,
                     help="Only commits after this date (e.g. 2022-01-01)")
    sc.add_argument("--before", default=None,
                     help="Only commits before this date (e.g. 2025-10-01)")
    sc.add_argument("--ext", nargs="*", default=None,
                     help="File extensions to include (e.g. .py .js). Default: all")
    sc.add_argument("--max-commits", type=int, default=0,
                     help="Max commits per repo (0 = all)")
    sc.add_argument("--include", default=None,
                     help="Only scan repos in this list file (owner/repo, one per line)")
    sc.add_argument("--exclude", default=None,
                     help="Skip repos in this list file (owner/repo, one per line)")
    sc.set_defaults(func=cmd_scan)

    # langs
    lg = sub.add_parser("langs", help="File extension stats for AI vs human commits")
    lg.add_argument("--repos", required=True, help="Directory of cloned repos")
    lg.add_argument("--after", default=None, help="Only commits after this date")
    lg.set_defaults(func=cmd_langs)

    # stats
    s = sub.add_parser("stats", help="Stats for JSONL or diffs dir")
    s.add_argument("--input", required=True, help="JSONL file or diffs directory")
    s.set_defaults(func=cmd_stats)

    # export
    e = sub.add_parser("export", help="Export diffs to train/test parquet")
    e.add_argument("--input", required=True,
                    help="Diffs directory from scan step")
    e.add_argument("--output", default="export/",
                    help="Output directory for parquet files (default: export/)")
    e.add_argument("--human-before", default="2022-12-31",
                    help="Only include human commits before this date (default: 2022-12-31)")
    e.add_argument("--test-split", type=float, default=0.1,
                    help="Fraction for test split (default: 0.1)")
    e.add_argument("--max-human-per-repo", type=int, default=None,
                    help="Cap human commits per repo for diversity")
    e.add_argument("--repo-split", action="store_true",
                    help="Split by repo (no repo in both train and test)")
    e.add_argument("--save-split", default=None,
                    help="Save test repo list to file")
    e.add_argument("--load-split", default=None,
                    help="Load test repo list from file (overrides --test-split)")
    e.add_argument("--seed", type=int, default=42, help="Random seed")
    e.set_defaults(func=cmd_export)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
