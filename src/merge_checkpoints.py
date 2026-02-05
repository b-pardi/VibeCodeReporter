"""
Merge individual repo checkpoint CSVs into a single combined dataset.

Usage:
    python src/merge_checkpoints.py
    python src/merge_checkpoints.py --checkpoint-dir output/repo_checkpoints --output output/combined_data.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def merge_checkpoints(checkpoint_dir: str, output_path: str, verbose: bool = True):
    """
    Merge all checkpoint CSVs from a directory into a single CSV.

    Args:
        checkpoint_dir: Directory containing per-repo checkpoint CSVs
        output_path: Path for the merged output CSV
        verbose: Print progress information
    """
    checkpoint_path = Path(checkpoint_dir)
    output_file = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find all checkpoint CSVs
    csv_files = list(checkpoint_path.glob("repo_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No checkpoint CSVs found in {checkpoint_dir}")

    if verbose:
        print(f"Found {len(csv_files)} checkpoint files")

    # Read and concatenate all CSVs
    dfs = []
    total_rows = 0

    iterator = tqdm(csv_files, desc="Merging", unit="file") if verbose else csv_files

    for csv_file in iterator:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            dfs.append(df)
            total_rows += len(df)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to read {csv_file.name}: {e}")

    if not dfs:
        raise ValueError("No valid checkpoint files could be read")

    # Concatenate all dataframes
    if verbose:
        print(f"Concatenating {len(dfs)} dataframes ({total_rows} total rows)...")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Save merged output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False, encoding='utf-8', escapechar="\\")

    if verbose:
        print(f"Merged CSV saved to: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")

    return combined_df


def main():
    parser = argparse.ArgumentParser(description='Merge checkpoint CSVs into single dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='output/repo_checkpoints',
                        help='Directory containing checkpoint CSVs')
    parser.add_argument('--output', type=str, default='output/combined_data.csv',
                        help='Output path for merged CSV')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    merge_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
