"""
Build a unified CSV dataset from all JSON repo files.
Adds columns for domain, language, and is_pre_2022.
"""

import json
import csv
import os
from pathlib import Path


def parse_filename(filename: str) -> tuple[str, str]:
    """Extract domain and language from filename like 'compiler_python_repos.json'."""
    # Remove _repos.json suffix
    base = filename.replace("_repos.json", "")
    # Split on last underscore to handle cases like 'machine-learning_python'
    parts = base.rsplit("_", 1)
    if len(parts) == 2:
        domain, language = parts
        return domain, language
    return base, "unknown"


def flatten_repo(repo: dict) -> dict:
    """Flatten nested repo data, extracting owner info to top level."""
    flat = {}
    for key, value in repo.items():
        if key == "owner" and isinstance(value, dict):
            # Flatten owner fields with 'owner_' prefix
            for owner_key, owner_value in value.items():
                flat[f"owner_{owner_key}"] = owner_value
        elif key == "license" and isinstance(value, dict):
            # Flatten license fields with 'license_' prefix
            for lic_key, lic_value in value.items():
                flat[f"license_{lic_key}"] = lic_value
        elif isinstance(value, (dict, list)):
            # Convert complex types to JSON string
            flat[key] = json.dumps(value)
        else:
            flat[key] = value
    return flat


def build_dataset(data_dir: Path, output_path: Path):
    """Build CSV dataset from all JSON files in pre2022 and post2022 directories."""
    all_repos = []
    all_fields = set()

    for time_period in ["pre2022", "post2022"]:
        period_dir = data_dir / time_period
        if not period_dir.exists():
            print(f"Warning: {period_dir} does not exist")
            continue

        for json_file in period_dir.glob("*.json"):
            domain, language = parse_filename(json_file.name)
            is_pre_2022 = time_period == "pre2022"

            print(f"Processing {json_file.name} (domain={domain}, lang={language}, pre2022={is_pre_2022})")

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data.get("items", [])
            for repo in items:
                flat_repo = flatten_repo(repo)
                flat_repo["domain"] = domain
                flat_repo["language"] = language
                flat_repo["is_pre_2022"] = is_pre_2022
                all_fields.update(flat_repo.keys())
                all_repos.append(flat_repo)

    # Sort fields for consistent column order, with added columns first
    priority_fields = ["domain", "language", "is_pre_2022", "id", "name", "full_name", "html_url", "description"]
    other_fields = sorted(all_fields - set(priority_fields))
    fieldnames = [f for f in priority_fields if f in all_fields] + other_fields

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for repo in all_repos:
            writer.writerow(repo)

    print(f"\nDataset created: {output_path}")
    print(f"Total repos: {len(all_repos)}")
    print(f"Total columns: {len(fieldnames)}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "github_repos"
    output_path = project_root / "data" / "repos_dataset.csv"

    build_dataset(data_dir, output_path)
