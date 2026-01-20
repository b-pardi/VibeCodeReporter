import argparse

from src.mining import mine_repos

# this is very much a placeholder but fine for testing the indv components of the project

def parse_args():
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        '-f', '--find_repos',
        action='store_true',
        help="Find Repos will search grab links to repos and groups them into domains to later be mined"
    )
    mode_group.add_argument(
        '-m', '--mine_repos',
        action='store_true',
        help="Read from `data/repo_links` and extracts diff files and metadata from all (filtered) commits."
    )
    mode_group.add_argument(
        '-d', '--detect_code_writer',
        action='store_true',
        help="Detect ML vs Human written code from project-commit diff files"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.find_repos:
        print("Work in Progress")
    elif args.mine_repos:
        mine_repos()
    elif args.detect_code_writer:
        print("Not yet implemented")

if __name__ == '__main__':
    main()