from pydriller import Repository
import pandas as pd
import datetime
import os
from tqdm import tqdm

from src.configs import mining_cfg
from utils.mining_utils import parse_unified_diff, get_num_commits_from_repo, read_repo_urls_file, is_code_file


# TODO: may want to iterate over `x` commits at a time instead of one to balance memory and requests time


def extract_info_from_repo(repo_url):
    """
    Yield row dictionaries, one per modified file per commit.
    This generator streams results rather than holding everything in memory

    args:
        repo_url: str = string to a github repository
    """

    # PyDriller's Repository(...) iterator yields Commit objects
    # can filter by date range, branch, etc.
    repository_iterator = Repository(
        path_to_repo=repo_url,
        since=mining_cfg.date_since,
        to=mining_cfg.date_to,
        only_in_main_branch=mining_cfg.only_in_main_branch,
    ).traverse_commits()

    for commit in repository_iterator:
        

        # commit.modified_files is a list of ModifiedFile objects.
        for modified_file in commit.modified_files:
            # NOTE: old_path/new_path can be None in some edge cases (e.g., deletions or weird renames)
            row = {
                # Repo identity
                "repo": str(repo_url),

                # commit metadata
                "commit_hash": commit.hash,
                "commit_author_name": commit.author.name,
                "commit_author_email": commit.author.email,
                "commit_author_date": commit.author_date.isoformat(),
                "commit_committer_name": commit.committer.name,
                "commit_committer_email": commit.committer.email,
                "commit_commit_date": commit.commit_date.isoformat(),
                "commit_message": commit.msg,

                # file info
                "file_old_path": modified_file.old_path,
                "file_new_path": modified_file.new_path,
                "file_change_type": str(modified_file.change_type),

                # line counts
                "added_lines": modified_file.added_lines,
                "deleted_lines": modified_file.deleted_lines,

                # language detected from filename extension
                "language": modified_file.language,

                # diff text (necessary for our project to feed to model)
                "diff": modified_file.diff if mining_cfg.include_diff else None,
            }

            yield row

def mine_repos():
    """ ENTRY POINT
    Traverse repositories and return a pandas DataFrame
    with one row per modified file per commit.
    """
    repo_urls_list = read_repo_urls_file(mining_cfg.repo_urls_file_path)
    rows = []
    # iterating over repositories
    for repo_url in tqdm(repo_urls_list, desc="Repositories", unit="repo"):
        repo_name = os.path.basename(repo_url.rstrip("/"))

        commit_iterator = Repository( # pydriller generator to iterate over indv commits in a repo
            path_to_repo=repo_url,
            since=mining_cfg.date_since,
            to=mining_cfg.date_to,
            only_no_merge=mining_cfg.only_no_merge,
            only_in_branch=mining_cfg.only_in_branch,
            only_releases=mining_cfg.only_releases,
            num_workers=mining_cfg.num_workers
        ).traverse_commits()

        # progress bar over commits
        for commit in tqdm(commit_iterator, total=get_num_commits_from_repo(repo_url), desc=f"Commits ({repo_name})", unit="commit", leave=False):
            if commit.merge: # skip merge commits
                continue

            for modified_file in commit.modified_files:
                diff_text = modified_file.diff # skip files with no changed text
                if not diff_text:
                    continue
                
                if mining_cfg.only_code_files and not is_code_file(modified_file): # optionally filter for only code files
                    continue

                changed_code = parse_unified_diff(diff_text)

                # Skip trivial snippets
                if len(changed_code) < 10:
                    continue

                rows.append({
                    "repo": repo_url,

                    # Commit metadata
                    "commit_hash": commit.hash,
                    "author_name": commit.author.name,
                    "author_email": commit.author.email,
                    "author_date": commit.author_date,
                    "commit_date": commit.committer_date,
                    "commit_message": commit.msg,

                    # File metadata
                    "file_old_path": modified_file.old_path,
                    "file_new_path": modified_file.new_path,
                    "change_type": str(modified_file.change_type),
                    "nloc": modified_file.nloc,
                    "complexity": modified_file.complexity,
                    "token_count": modified_file.token_count,
                    "source_code": modified_file.source_code if mining_cfg.include_src_code else None, # optionally include src code

                    # line counts
                    "added_lines": modified_file.added_lines,
                    "deleted_lines": modified_file.deleted_lines,

                    # code unit; could also do removed or added only
                    "code_unit_type": "changed",
                    "code_unit_text": changed_code,
                    "code_unit_char_len": len(changed_code),
                })

    repos_df = pd.DataFrame(rows)
    print(repos_df.head())
    repos_df.to_csv(mining_cfg.output_csv_path, index=False, encoding='utf-8', escapechar="\\",)
    print(f"COMPLETED: Saved csv to: {mining_cfg.output_csv_path}")
    return repos_df