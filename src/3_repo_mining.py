"""
Extract code from GitHub repos using PyDriller for AI detection analysis.

For each repo, iterates through branches with sufficient commits,
extracts full file content and diffs, computes metrics, and saves per-repo CSV checkpoints.
Parallelizes at the commit level for optimal progress tracking.
"""

from pydriller import Repository
from pydriller.git import Git
import pandas as pd
import os
import sys
import re
import argparse
import queue
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback
from typing import NamedTuple, Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore", message=".*deallocator.*")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import MiningConfig, mining_cfg
from utils.mining_utils import parse_unified_diff, is_code_file
from utils.code_metrics import (
    compute_loc,
    compute_comment_density,
    compute_boilerplate_score,
    compute_complexity_metrics,
)
from utils.file_filters import FileFilter, FilterConfig
from utils.file_classifier import classify_file_type


class QueuedRepo(NamedTuple):
    """A repo that has been cloned and is ready for processing."""
    repo_id: str
    repo_url: str
    repo_name: str
    local_path: Path
    metadata: Dict[str, Any]


class CloneStats:
    """Thread-safe statistics for clone operations."""
    def __init__(self):
        self.lock = threading.Lock()
        self.cloned = 0
        self.reused = 0  # already cloned, reused from temp dir
        self.failed = 0
        self.queued = 0
        self.skipped_size = 0
        self.skipped_exists = 0
        self.current_cloning = ""

    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def increment(self, field: str, amount: int = 1):
        with self.lock:
            setattr(self, field, getattr(self, field) + amount)


def clone_repo(repo_url: str, dest_path: Path, timeout: int = 600) -> bool:
    """
    Clone a repo to a local path using git.
    Returns True if successful, False otherwise.
    """
    try:
        # Remove existing directory if it exists
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Clone with depth=1 initially, then fetch full history
        # Actually, for PyDriller we need full history, so do a full clone
        result = subprocess.run(
            ['git', 'clone', '--quiet', repo_url, str(dest_path)],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Clean up partial clone
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)
        return False
    except Exception:
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)
        return False


def delete_cloned_repo(local_path: Path) -> bool:
    """Delete a cloned repo directory. Non-blocking - uses background process."""
    import gc

    if not local_path.exists():
        return True

    # Force garbage collection to release any GitPython file handles
    gc.collect()

    try:
        # Use background subprocess for both Windows and Linux to avoid blocking
        if sys.platform == 'win32':
            subprocess.Popen(
                f'cmd /c rd /s /q "{local_path}" >nul 2>&1',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # Linux/WSL: use rm -rf in background
            subprocess.Popen(
                f'rm -rf "{local_path}" 2>/dev/null &',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True  # Assume success, don't block
    except Exception:
        return False


def cloner_worker(
    repo_queue: queue.Queue,
    clone_queue: queue.Queue,
    cfg: 'MiningConfig',
    clone_stats: CloneStats,
    stop_event: threading.Event,
):
    """
    Worker thread that clones repos from repo_queue and puts them in clone_queue.
    Runs until stop_event is set and repo_queue is empty.
    """
    while not stop_event.is_set() or not repo_queue.empty():
        try:
            # Get next repo to clone (with timeout to check stop_event)
            try:
                repo_info = repo_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if repo_info is None:  # Sentinel value
                repo_queue.task_done()
                break

            repo_id = str(repo_info['id'])
            repo_url = repo_info['clone_url']
            repo_name = repo_info.get('name', repo_id)
            repo_size_kb = repo_info.get('size', 0)
            repo_size_mb = repo_size_kb / 1024 if repo_size_kb else 0

            clone_stats.update(current_cloning=repo_name[:20])

            # Check size limit
            if cfg.max_repo_size_mb and repo_size_mb > cfg.max_repo_size_mb:
                clone_stats.increment('skipped_size')
                repo_queue.task_done()
                continue

            # Check if already processed (checkpoint exists)
            if is_repo_processed(cfg, repo_id):
                clone_stats.increment('skipped_exists')
                repo_queue.task_done()
                continue

            # Check if already cloned but not yet processed
            local_path = cfg.clone_repo_to / f"repo_{repo_id}"
            if local_path.exists() and (local_path / '.git').exists():
                # Already cloned, just queue it
                queued_repo = QueuedRepo(
                    repo_id=repo_id,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    local_path=local_path,
                    metadata=dict(repo_info),
                )
                clone_queue.put(queued_repo)
                clone_stats.increment('reused')
                clone_stats.increment('queued')
                repo_queue.task_done()
                continue

            # Clone the repo
            success = clone_repo(repo_url, local_path)

            if success:
                # Put in clone queue (blocks if queue is full)
                # Commit count check happens in processor via get_branches_with_commit_counts
                queued_repo = QueuedRepo(
                    repo_id=repo_id,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    local_path=local_path,
                    metadata=dict(repo_info),
                )
                clone_queue.put(queued_repo)
                clone_stats.increment('cloned')
                clone_stats.increment('queued')
            else:
                clone_stats.increment('failed')

            repo_queue.task_done()

        except Exception as e:
            # Don't crash the thread on errors
            continue

    clone_stats.update(current_cloning="")


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def estimate_row_size(row: dict) -> int:
    """Estimate the size of a row in bytes when serialized to CSV."""
    size = 0
    for value in row.values():
        if value is None:
            size += 0
        elif isinstance(value, str):
            size += len(value.encode('utf-8', errors='replace'))
        else:
            size += len(str(value))
    return size


def build_file_filter(cfg: MiningConfig) -> FileFilter:
    """Build FileFilter from MiningConfig."""
    filter_config = FilterConfig(
        min_size=cfg.min_file_size,
        max_size=cfg.max_file_size,
        min_lines=cfg.min_lines,
        max_line_length=cfg.max_line_length,
        exclude_autogenerated=cfg.exclude_autogenerated,
        exclude_vendored=cfg.exclude_vendored,
        exclude_tests=cfg.exclude_tests,
        exclude_minified=cfg.exclude_minified,
        max_comment_ratio=cfg.max_comment_ratio,
        excluded_paths=cfg.excluded_paths,
        autogen_markers=cfg.autogen_markers,
    )
    return FileFilter(filter_config)


def get_checkpoint_path(cfg: MiningConfig, repo_id: str) -> Path:
    """Get path to checkpoint pickle file for a repo."""
    safe_id = re.sub(r'[^\w\-_.]', '_', str(repo_id))
    return cfg.checkpoint_dir / f"repo_{safe_id}.pkl"


def is_repo_processed(cfg: MiningConfig, repo_id: str) -> bool:
    """Check if repo has already been processed (checkpoint exists)."""
    return get_checkpoint_path(cfg, repo_id).exists()


def save_checkpoint(cfg: MiningConfig, repo_id: str, rows: list) -> Path:
    """Save checkpoint pickle file for a repo."""
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = get_checkpoint_path(cfg, repo_id)
    df = pd.DataFrame(rows)
    df.to_pickle(checkpoint_path)
    return checkpoint_path


def get_repo_total_commits(repo_url: str, cfg: MiningConfig, debug: bool = False) -> int:
    """Get total commit count for a repo (with date filtering)."""
    try:
        count = 0
        for _ in Repository(
            path_to_repo=repo_url,
            since=cfg.date_since,
            to=cfg.date_to,
            only_no_merge=cfg.only_no_merge,
        ).traverse_commits():
            count += 1
        if debug:
            print(f"  DEBUG: {repo_url} has {count} commits (date range: {cfg.date_since} to {cfg.date_to})")
        return count
    except Exception as e:
        if debug:
            print(f"  DEBUG: Error counting commits for {repo_url}: {e}")
        return 0


def get_branches_with_commit_counts(repo_url: str, cfg: MiningConfig) -> list:
    """
    Get list of branches with their commit counts.
    Returns list of (branch_name, commit_count) tuples.
    """
    branches = []
    try:
        git_repo = Git(repo_url)
        repo_branches = git_repo.repo.branches

        for branch in repo_branches:
            branch_name = branch.name
            try:
                count = 0
                for _ in Repository(
                    path_to_repo=repo_url,
                    since=cfg.date_since,
                    to=cfg.date_to,
                    only_no_merge=cfg.only_no_merge,
                    only_in_branch=branch_name,
                ).traverse_commits():
                    count += 1

                if count >= cfg.min_branch_commits:
                    branches.append((branch_name, count))
            except Exception:
                continue

    except Exception:
        # Fallback: just use default branch
        try:
            count = get_repo_total_commits(repo_url, cfg)
            if count >= cfg.min_branch_commits:
                branches.append((None, count))
        except Exception:
            pass

    return branches


def process_commit(commit, repo_id: str, repo_url: str, branch_name: str,
                   cfg: MiningConfig, file_filter: FileFilter,
                   worker_stats: dict = None) -> list:
    """
    Process a single commit and return list of row dicts.
    This function is designed to be called in parallel.
    """
    # Track worker activity
    if worker_stats is not None:
        thread_id = threading.current_thread().name
        with worker_stats['lock']:
            worker_stats['active'] += 1
            if thread_id not in worker_stats['jobs_per_worker']:
                worker_stats['jobs_per_worker'][thread_id] = 0
            worker_stats['jobs_per_worker'][thread_id] += 1

    rows = []
    # Debug counters (use worker_stats lock for thread safety)
    filter_stats = {'no_diff': 0, 'not_code': 0, 'file_filter': 0, 'too_few_lines': 0, 'passed': 0}

    try:
        for modified_file in commit.modified_files:
            file_path = modified_file.new_path or modified_file.old_path or ""

            # Skip if no diff
            diff_text = modified_file.diff
            if not diff_text:
                filter_stats['no_diff'] += 1
                continue

            # Apply code file filter
            if cfg.only_code_files and not is_code_file(modified_file):
                filter_stats['not_code'] += 1
                continue

            # Get source code
            source_code = modified_file.source_code or ""

            # Apply file filters
            if source_code:
                filter_result = file_filter.check(file_path, source_code)
                if not filter_result.passed:
                    filter_stats['file_filter'] += 1
                    continue

            # Parse diff to get added lines
            added_code = parse_unified_diff(diff_text, only_additions=True)

            # Skip trivial changes (check line count, not character count)
            added_lines_count = added_code.count('\n') + 1 if added_code else 0
            if added_lines_count < cfg.min_lines:
                filter_stats['too_few_lines'] += 1
                continue

            filter_stats['passed'] += 1

            # Get language from file extension
            ext = os.path.splitext(file_path)[1].lower()
            ext_to_lang = {
                '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                '.java': 'Java', '.c': 'C', '.cpp': 'C++', '.cc': 'C++',
                '.h': 'C', '.hpp': 'C++', '.cs': 'C#', '.go': 'Go',
                '.rs': 'Rust', '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift',
                '.kt': 'Kotlin', '.scala': 'Scala', '.r': 'R', '.m': 'Objective-C',
                '.mm': 'Objective-C++', '.pl': 'Perl', '.sh': 'Shell', '.bash': 'Shell',
            }
            language = ext_to_lang.get(ext, ext[1:] if ext else "")

            # Compute metrics if enabled
            if cfg.include_metrics and source_code:
                loc_full = compute_loc(source_code)
                loc_diff = compute_loc(added_code)
                comment_density = compute_comment_density(source_code, language)
                boilerplate_score = compute_boilerplate_score(source_code, language)
                complexity = compute_complexity_metrics(source_code, language)
                file_type = classify_file_type(file_path, source_code)
            else:
                loc_full = modified_file.nloc or 0
                loc_diff = modified_file.added_lines
                comment_density = None
                boilerplate_score = None
                complexity = {}
                file_type = classify_file_type(file_path, "")

            row = {
                "repo_id": repo_id,
                "repo_url": repo_url,
                "branch": branch_name or "default",
                "commit_hash": commit.hash,
                "commit_date": commit.committer_date.isoformat(),
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "commit_message": commit.msg,
                "file_path": file_path,
                "file_old_path": modified_file.old_path,
                "file_type": file_type,
                "change_type": str(modified_file.change_type),
                "language": language,
                "full_code": source_code if cfg.include_src_code else None,
                "diff_added": added_code if cfg.include_diff else None,
                "loc_full": loc_full,
                "loc_diff": loc_diff,
                "added_lines": modified_file.added_lines,
                "deleted_lines": modified_file.deleted_lines,
                "comment_density": comment_density,
                "boilerplate_score": boilerplate_score,
                "cyclomatic_complexity": complexity.get('cyclomatic', modified_file.complexity),
                "cognitive_complexity": complexity.get('cognitive', None),
                "max_nesting_depth": complexity.get('max_nesting', None),
                "token_count": modified_file.token_count,
                "snippets_fixed_lines": None,
                "snippets_functions": None,
                "snippets_classes": None,
            }
            rows.append(row)

    except Exception as e:
        # Silently skip problematic commits (GitPython parsing errors are common)
        pass
    finally:
        # Update worker stats and aggregate filter stats
        if worker_stats is not None:
            with worker_stats['lock']:
                worker_stats['active'] -= 1
                worker_stats['completed'] += 1
                # Aggregate filter stats
                if 'filter_stats' not in worker_stats:
                    worker_stats['filter_stats'] = {'no_diff': 0, 'not_code': 0, 'file_filter': 0, 'too_few_lines': 0, 'passed': 0}
                for k, v in filter_stats.items():
                    worker_stats['filter_stats'][k] += v

    return rows


def filter_commits_by_sampling(commits: list, cfg: MiningConfig) -> list:
    """
    Filter commits by skip interval and time interval.
    Returns filtered list of commits.
    """
    if not commits:
        return []

    filtered = []
    skip_interval = cfg.commit_skip if cfg.commit_skip > 1 else 1
    min_interval_seconds = cfg.min_commit_interval_days * 24 * 60 * 60
    last_grabbed_timestamp = None

    for i, commit in enumerate(commits):
        # Apply skip interval (every Nth)
        if (i + 1) % skip_interval != 0:
            continue

        # Apply time-based filtering
        if min_interval_seconds > 0:
            commit_ts = commit.committer_date.timestamp()
            if last_grabbed_timestamp is not None:
                if abs(last_grabbed_timestamp - commit_ts) < min_interval_seconds:
                    continue
            last_grabbed_timestamp = commit_ts

        filtered.append(commit)

    return filtered


def process_cloned_repo(
    queued_repo: QueuedRepo,
    cfg: MiningConfig,
    file_filter: FileFilter,
    worker_stats: dict,
    is_parallel: bool,
    total_bytes_ref: list,  # Using list as mutable reference
    branch_pbar_position: int = 1,
) -> list:
    """
    Process a single cloned repo and return list of row dicts.
    Uses local_path instead of repo_url for PyDriller operations.
    """
    repo_id = queued_repo.repo_id
    repo_url = queued_repo.repo_url  # Keep original URL for metadata
    local_path = str(queued_repo.local_path)
    repo_rows = []

    # Get branches from local clone
    branches = get_branches_with_commit_counts(local_path, cfg)
    if not branches:
        return repo_rows

    if is_parallel:
        # Parallel mode: use nested tqdm bars (branch bar + commit completion bar)
        branch_pbar = tqdm(
            branches,
            desc="    Branches",
            unit="branch",
            leave=False,
            position=branch_pbar_position,
        )

        for branch_name, branch_commit_count in branch_pbar:
            branch_display = (branch_name or "default")[:20]
            branch_pbar.set_postfix_str(branch_display)

            try:
                repository = Repository(
                    path_to_repo=local_path,
                    since=cfg.date_since,
                    to=cfg.date_to,
                    only_no_merge=cfg.only_no_merge,
                    only_in_branch=branch_name,
                    only_releases=cfg.only_releases,
                )

                # Collect commits
                all_commits = []
                for commit in repository.traverse_commits():
                    if not commit.merge:
                        all_commits.append(commit)

            except Exception:
                continue

            # Apply sampling
            filtered_commits = filter_commits_by_sampling(all_commits, cfg)
            if not filtered_commits:
                continue

            branch_pbar.set_postfix_str(f"{branch_display} ({len(filtered_commits)} commits)")

            # Process commits in parallel with progress bar
            commit_pbar = tqdm(
                total=len(filtered_commits),
                desc="      Commits",
                unit="commit",
                leave=False,
                position=branch_pbar_position + 1,
            )

            executor = ThreadPoolExecutor(max_workers=cfg.num_workers)
            timed_out_commits = []
            try:
                futures = {
                    executor.submit(
                        process_commit, commit, repo_id, repo_url,
                        branch_name, cfg, file_filter, worker_stats
                    ): commit.hash
                    for commit in filtered_commits
                }

                # Process with individual timeouts - don't let one stuck commit block others
                done_count = 0
                for future in as_completed(futures, timeout=cfg.batch_timeout):
                    commit_hash = futures[future]
                    try:
                        rows = future.result(timeout=cfg.commit_timeout)
                        for r in rows:
                            row_size = estimate_row_size(r)
                            total_bytes_ref[0] += row_size
                            repo_rows.append(r)
                    except TimeoutError:
                        timed_out_commits.append(commit_hash[:8])
                        future.cancel()
                    except Exception:
                        pass

                    done_count += 1
                    commit_pbar.update(1)
                    with worker_stats['lock']:
                        commit_pbar.set_postfix_str(
                            f"Est: {format_size(total_bytes_ref[0])} | Workers: {worker_stats['active']}/{cfg.num_workers}"
                        )

            except TimeoutError:
                # Batch timeout - find which commits are still pending
                pending = [futures[f][:8] for f in futures if not f.done()]
                print(f"      [TIMEOUT] Batch timeout after {cfg.batch_timeout}s, {len(pending)} commits stuck: {pending[:5]}...")
            finally:
                # Log timed out commits
                if timed_out_commits:
                    print(f"      [TIMEOUT] {len(timed_out_commits)} commits timed out: {timed_out_commits[:5]}...")
                # Shutdown executor without waiting for stuck workers
                executor.shutdown(wait=False, cancel_futures=True)
                # Reset worker count since cancelled workers don't decrement
                with worker_stats['lock']:
                    worker_stats['active'] = 0

            commit_pbar.close()
        branch_pbar.close()

    else:
        # Sequential mode: show all nested progress bars
        branch_pbar = tqdm(
            branches,
            desc="    Branches",
            unit="branch",
            leave=False,
            position=branch_pbar_position,
        )

        for branch_name, branch_commit_count in branch_pbar:
            branch_display = (branch_name or "default")[:20]
            branch_pbar.set_postfix_str(branch_display)

            try:
                repository = Repository(
                    path_to_repo=local_path,
                    since=cfg.date_since,
                    to=cfg.date_to,
                    only_no_merge=cfg.only_no_merge,
                    only_in_branch=branch_name,
                    only_releases=cfg.only_releases,
                )

                all_commits = []
                for commit in repository.traverse_commits():
                    if not commit.merge:
                        all_commits.append(commit)

            except Exception:
                continue

            filtered_commits = filter_commits_by_sampling(all_commits, cfg)
            if not filtered_commits:
                continue

            branch_pbar.set_postfix_str(f"{branch_display} ({len(filtered_commits)} commits)")

            commit_pbar = tqdm(
                filtered_commits,
                desc="      Commits",
                unit="commit",
                leave=False,
                position=branch_pbar_position + 1,
            )

            for commit in commit_pbar:
                commit_pbar.set_postfix_str(f"{commit.hash[:8]} | Est: {format_size(total_bytes_ref[0])}")

                rows = process_commit(
                    commit, repo_id, repo_url, branch_name, cfg, file_filter
                )
                for r in rows:
                    row_size = estimate_row_size(r)
                    total_bytes_ref[0] += row_size
                    repo_rows.append(r)

            commit_pbar.close()
        branch_pbar.close()

    return repo_rows


def mine_repos_from_metadata(
    metadata_csv: str,
    cfg: MiningConfig = None,
    limit: int = None,
    domain: str = None,
    language: str = None,
    repo_ids: list = None,
):
    """
    Mine repos using the metadata CSV from build_dataset.py.

    Uses a producer-consumer pattern:
    - A cloner thread continuously clones repos ahead of time (up to cfg.max_cloned_repos)
    - The main thread processes cloned repos
    - Repos are deleted after processing

    Parallelizes at the commit level within each repo.
    """
    if cfg is None:
        cfg = mining_cfg

    # Load metadata
    df = pd.read_csv(metadata_csv)


    # Apply filters
    if repo_ids:
        df = df[df['id'].isin(repo_ids)]
    if domain:
        df = df[df['domain'] == domain]
    if language:
        df = df[df['language'] == language]
    if limit:
        df = df.head(limit)

    is_parallel = cfg.num_workers > 1
    total_repos = len(df)

    print(f"Processing {total_repos} repos")
    print(f"Config: min_repo_commits={cfg.min_repo_commits}, min_branch_commits={cfg.min_branch_commits}")
    print(f"        min_commit_interval_days={cfg.min_commit_interval_days}, commit_skip={cfg.commit_skip}")
    print(f"        num_workers={cfg.num_workers} ({'parallel commits' if is_parallel else 'sequential commits'})")
    print(f"        parallel_cloning={cfg.parallel_cloning} (clone while processing)")
    print(f"        max_cloned_repos={cfg.max_cloned_repos} (clone queue size)")
    if cfg.max_repo_size_mb:
        print(f"        max_repo_size_mb={cfg.max_repo_size_mb}")
    print()

    # Ensure directories exist
    cfg.clone_repo_to.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")

    # Build file filter
    file_filter = build_file_filter(cfg)

    # Track totals (using list as mutable reference for nested functions)
    total_bytes_ref = [0]
    successful = 0
    failed = 0
    skipped_commits = 0
    skipped_no_files = 0

    # Worker stats for parallel mode
    worker_stats = {
        'lock': threading.Lock(),
        'active': 0,
        'completed': 0,
        'jobs_per_worker': {},
    }

    # Clone stats
    clone_stats = CloneStats()

    # Create queues
    # repo_queue: repos waiting to be cloned (unbounded, filled once)
    # clone_queue: cloned repos waiting to be processed (bounded)
    repo_queue = queue.Queue()
    clone_queue = queue.Queue(maxsize=cfg.max_cloned_repos)

    # Fill repo_queue with all repos to process
    for _, row in df.iterrows():
        repo_queue.put(dict(row))

    # Add sentinel to signal end
    repo_queue.put(None)

    # Stop event for cloner thread
    stop_event = threading.Event()

    # Start cloner thread
    cloner_thread = threading.Thread(
        target=cloner_worker,
        args=(repo_queue, clone_queue, cfg, clone_stats, stop_event),
        daemon=True,
        name="ClonerThread",
    )
    cloner_thread.start()

    # Process repos from clone_queue
    processed = 0
    repo_pbar = tqdm(
        total=total_repos,
        desc="Repos",
        unit="repo",
        position=0,
    )

    def get_status():
        status = f"Est: {format_size(total_bytes_ref[0])}"
        with clone_stats.lock:
            status += f" | Cloned: {clone_stats.cloned}, Queued: {clone_queue.qsize()}"
            if clone_stats.current_cloning:
                status += f", Cloning: {clone_stats.current_cloning}"
        if is_parallel:
            with worker_stats['lock']:
                status += f" | Workers: {worker_stats['active']}/{cfg.num_workers}"
        return status

    while True:
        repo_pbar.set_postfix_str(get_status())

        # Check how many were skipped by cloner so far
        with clone_stats.lock:
            total_skipped = (clone_stats.skipped_size +
                           clone_stats.skipped_exists +
                           clone_stats.failed)

        # Try to get next cloned repo
        try:
            queued_repo = clone_queue.get(timeout=1.0)
        except queue.Empty:
            # Check if cloner is done and queue is empty
            if not cloner_thread.is_alive() and clone_queue.empty():
                break
            continue

        if queued_repo is None:
            # Sentinel from cloner
            break

        with clone_stats.lock:
            clone_stats.queued -= 1

        repo_pbar.set_description(f"Repo: {queued_repo.repo_name[:25]}")
        repo_pbar.set_postfix_str(get_status())

        try:
            # Quick commit count check
            local_path = str(queued_repo.local_path)
            total_commits = get_repo_total_commits(local_path, cfg, debug=False)
            if total_commits < cfg.min_repo_commits:
                repo_pbar.set_postfix_str(f"Skipped ({total_commits}<{cfg.min_repo_commits} commits)")
                delete_cloned_repo(queued_repo.local_path)
                skipped_commits += 1
                processed += 1
                repo_pbar.update(1)
                continue

            # Process the cloned repo
            repo_rows = process_cloned_repo(
                queued_repo=queued_repo,
                cfg=cfg,
                file_filter=file_filter,
                worker_stats=worker_stats,
                is_parallel=is_parallel,
                total_bytes_ref=total_bytes_ref,
                branch_pbar_position=1,
            )

            # Save checkpoint
            with worker_stats['lock']:
                fs = worker_stats.get('filter_stats', {})
                print(f"  [DEBUG] Repo {queued_repo.repo_name}: {len(repo_rows)} rows | "
                      f"no_diff={fs.get('no_diff',0)}, not_code={fs.get('not_code',0)}, "
                      f"file_filter={fs.get('file_filter',0)}, too_few_lines={fs.get('too_few_lines',0)}, "
                      f"passed={fs.get('passed',0)}")
                # Reset filter stats for next repo
                worker_stats['filter_stats'] = {'no_diff': 0, 'not_code': 0, 'file_filter': 0, 'too_few_lines': 0, 'passed': 0}
            if repo_rows:
                save_checkpoint(cfg, queued_repo.repo_id, repo_rows)
                successful += 1
                repo_pbar.set_postfix_str(f"Saved {len(repo_rows)} rows | {get_status()}")
            else:
                skipped_no_files += 1
                repo_pbar.set_postfix_str(f"No valid files | {get_status()}")

        except Exception as e:
            repo_pbar.set_postfix_str(f"Error: {str(e)[:30]} | {get_status()}")
            failed += 1

        finally:
            # Always delete the cloned repo after processing
            delete_cloned_repo(queued_repo.local_path)

        processed += 1
        repo_pbar.update(1)

        # Update progress to account for skipped repos
        with clone_stats.lock:
            new_total_skipped = (clone_stats.skipped_size +
                               clone_stats.skipped_exists +
                               clone_stats.failed)
            if new_total_skipped > total_skipped:
                skip_delta = new_total_skipped - total_skipped
                repo_pbar.update(skip_delta)
                total_skipped = new_total_skipped

    # Signal cloner to stop and wait for it
    stop_event.set()
    cloner_thread.join(timeout=5.0)

    # Final update for any remaining skipped
    with clone_stats.lock:
        final_skipped = clone_stats.skipped_size + clone_stats.skipped_exists
        final_failed = clone_stats.failed

    # Update progress bar to completion
    remaining = total_repos - repo_pbar.n
    if remaining > 0:
        repo_pbar.update(remaining)

    repo_pbar.close()

    # Print worker stats summary for parallel mode
    if is_parallel:
        print(f"\nWorker Statistics:")
        with worker_stats['lock']:
            for worker_name, job_count in sorted(worker_stats['jobs_per_worker'].items()):
                print(f"  {worker_name}: {job_count} commits processed")
            print(f"  Total commits processed: {worker_stats['completed']}")

    # Print clone stats
    print(f"\nClone Statistics:")
    with clone_stats.lock:
        print(f"  Successfully cloned: {clone_stats.cloned}")
        print(f"  Reused from temp dir: {clone_stats.reused}")
        print(f"  Failed to clone: {clone_stats.failed}")
        print(f"  Skipped (size): {clone_stats.skipped_size}")
        print(f"  Skipped (already processed): {clone_stats.skipped_exists}")

    print(f"\nProcessing Statistics:")
    print(f"  Saved checkpoints: {successful}")
    print(f"  Skipped (too few commits): {skipped_commits}")
    print(f"  Skipped (no valid files): {skipped_no_files}")
    print(f"  Failed: {failed}")

    print(f"\n{'='*60}")
    print(f"Total estimated dataset size: {format_size(total_bytes_ref[0])}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    if successful > 0:
        print(f"Run 'python src/merge_checkpoints.py' to combine into single CSV")


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Mine code from GitHub repos using PyDriller')

    # Output
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='output/repo_checkpoints',
                        help='Directory for per-repo checkpoint CSVs')

    # Date filters
    parser.add_argument('--since', type=str, help='Only commits after this date (YYYY-MM-DD)')
    parser.add_argument('--to', type=str, help='Only commits before this date (YYYY-MM-DD)')

    # Repo/branch thresholds
    parser.add_argument('--min-repo-commits', type=int, default=None,
                        help='Min commits for a repo to be considered (default: from config)')
    parser.add_argument('--min-branch-commits', type=int, default=None,
                        help='Min commits for a branch to be analyzed (default: from config)')
    parser.add_argument('--max-repo-size-mb', type=int, default=None,
                        help='Skip repos larger than this size in MB (default: no limit)')

    # Commit sampling (None = use config value)
    parser.add_argument('--min-commit-interval-days', type=int, default=None,
                        help='Min days between grabbed commits (0 = grab all)')
    parser.add_argument('--commit-skip', type=int, default=None,
                        help='Take every Nth commit (0 or 1 = no skip)')

    # File filters
    parser.add_argument('--min-file-size', type=int, default=500, help='Min file size in bytes')
    parser.add_argument('--max-file-size', type=int, default=100000, help='Max file size in bytes')
    parser.add_argument('--min-lines', type=int, default=10, help='Min lines of code')

    # Content filters
    parser.add_argument('--include-autogenerated', action='store_true')
    parser.add_argument('--include-vendored', action='store_true')
    parser.add_argument('--exclude-tests', action='store_true')

    # Parallelization
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of threads for parallel commit processing (default: 4, use 1 for sequential)')
    parser.add_argument('--max-cloned-repos', type=int, default=10,
                        help='Max repos to clone ahead of processing (default: 10)')

    # Subset selection
    parser.add_argument('--repo-ids', type=int, nargs='+', help='Specific repo IDs to process')
    parser.add_argument('--limit', type=int, help='Limit number of repos')
    parser.add_argument('--domain', type=str, help='Filter by domain')
    parser.add_argument('--language', type=str, help='Filter by language')

    args = parser.parse_args()

    # Build config
    cfg = MiningConfig()

    if args.output_dir:
        cfg.output_base_path = Path(args.output_dir)
        cfg.output_csv_path = cfg.output_base_path / 'repo_data.csv'
    if args.checkpoint_dir:
        cfg.checkpoint_dir = Path(args.checkpoint_dir)
    if args.since:
        cfg.date_since = datetime.strptime(args.since, '%Y-%m-%d')
    if args.to:
        cfg.date_to = datetime.strptime(args.to, '%Y-%m-%d')
    if args.max_repo_size_mb:
        cfg.max_repo_size_mb = args.max_repo_size_mb

    if args.min_repo_commits is not None:
        cfg.min_repo_commits = args.min_repo_commits
    if args.min_branch_commits is not None:
        cfg.min_branch_commits = args.min_branch_commits
    if args.min_commit_interval_days is not None:
        cfg.min_commit_interval_days = args.min_commit_interval_days
    if args.commit_skip is not None:
        cfg.commit_skip = args.commit_skip
    cfg.min_file_size = args.min_file_size
    cfg.max_file_size = args.max_file_size
    cfg.min_lines = args.min_lines
    cfg.num_workers = args.num_workers
    cfg.max_cloned_repos = args.max_cloned_repos

    if args.include_autogenerated:
        cfg.exclude_autogenerated = False
    if args.include_vendored:
        cfg.exclude_vendored = False
    if args.exclude_tests:
        cfg.exclude_tests = True

    mine_repos_from_metadata(
        metadata_csv=cfg.metadata_csv,
        cfg=cfg,
        limit=args.limit,
        domain=args.domain,
        language=args.language,
        repo_ids=args.repo_ids,
    )


if __name__ == '__main__':
    main()
