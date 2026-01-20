from dataclasses import dataclass
from typing import Optional, List, Iterable
from pathlib import Path
from datetime import datetime

@dataclass()
class MiningConfig:
    """
    Controls what we mine and where we write it.
    """
    # File Paths:
    data_base_path: Path = Path("data").resolve()
    output_base_path: Path = Path("output").resolve()
    repo_urls_file_path: Path = data_base_path / 'repo_links.txt'   # path to file containing git repos to drill
    output_csv_path: Path = output_base_path / 'repo_data.csv'      # path to output file from pydriller
    clone_repo_to: Path = data_base_path / 'temp'                   # path to where pydriller stores temporary commits

    # Filters:
    date_since: str = "2020-01-01"          # ISO date string formatted as "YYYY-mm-dd" indicating earliest date to grab commits from
    date_to: str = "2025-12-31"             # ISO date string formatted as "YYYY-mm-dd" indicating latest date to grab commits from
    only_in_main_branch: bool = False       # only analyze commits from main
    only_no_merge: bool = True              # only analyze commits that are not merges
    only_in_branch: Optional[str] = None    # only analyze commits in this branch (e.g. main)
    only_releases: bool = False             # only analyze commits that from 'release' versions
    only_code_files: bool = True            # whitelists files based on extensions listed in `consts/code_file_exts.py`
    min_lines: int = 10                     # min number of lines changed to skip trivial commits
    min_commits: int = 10                   # min number of commits for a repo to have to consider for analysis        
    include_diff: bool = True               # include the diff of files in each commit
    include_src_code: bool = False          # include the src code of files in each commit

    # misc  
    num_workers: int = 1                    # number of workers/threads; note: if num_workers > 1 the commits order is not maintained.


# global config objects
mining_cfg = MiningConfig()

# format time strings to datetime objects (easier with pandas)
mining_cfg.date_since = datetime.strptime(mining_cfg.date_since, '%Y-%m-%d')
mining_cfg.date_to = datetime.strptime(mining_cfg.date_to, '%Y-%m-%d')