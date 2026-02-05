#! util fns pertaining to the pydriller git mining operations
from pydriller import Repository
from pydriller.git import Git
import os

from src.configs import mining_cfg
from consts.code_file_exts import CODE_EXTENSIONS, ALT_FNS


def parse_unified_diff(unified_diff_str, only_additions=False):
    """
    Convert a unified diff patch into just the added lines of code.

    Rules:
    - Keep only lines that start with '+' or '-'
    - Ignores diff metadata lines: '+++ b/file', '+++' headers, etc.
    - Ignores hunk headers '@@ ... @@'
    - Remove the leading '+'/'-' so the output is plain code text
    """
    changed_code_lines = []
    for line in unified_diff_str.splitlines():
        # skip file and hunk headers
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue

        # keep added lines only if only_additions
        if line.startswith("+"):
            # excluded the +++ header above
            changed_code_lines.append(line)

        if line.startswith("-") and only_additions is False:
            changed_code_lines.append(line)


    return "\n".join(changed_code_lines).strip()

def get_num_commits_from_repo(repo_url, filter=True):
    """Get number of commits from a repository and optionally applies filter spec'd in config"""
    if not filter:
        git_repo = Git(repo_url)
        return git_repo.total_commits()
    else:
        return len(list(iter(Repository( # pydriller generator to iterate over indv commits in a repo
            path_to_repo=repo_url,
            since=mining_cfg.date_since,
            to=mining_cfg.date_to,
            only_no_merge=mining_cfg.only_no_merge,
            only_in_branch=mining_cfg.only_in_branch,
            only_releases=mining_cfg.only_releases
        ).traverse_commits())))

def read_repo_urls_file(fp):
    if os.path.getsize(fp) == 0:
        raise Exception(f"ERROR: Found no valid links in {fp}")
    
    repo_urls_list = []
    with open(fp, 'r') as repos_file:
        for line in repos_file:
            url = line.strip()
            if not url.endswith('.git'):
                raise Exception(f"ERROR: Found invalid URL: {url} in {fp}\nURLs must end in '.git'")
            repo_urls_list.append(url)

    return repo_urls_list

def is_code_file(modified_file):
    """Check if a modified file is a code file based on extension."""
    path_lower = (modified_file.new_path or modified_file.old_path or "").lower()
    fn = os.path.basename(path_lower)
    _, ext = os.path.splitext(path_lower)

    # Return True if extension is in CODE_EXTENSIONS or filename is in ALT_FNS
    return ext in CODE_EXTENSIONS or fn in ALT_FNS