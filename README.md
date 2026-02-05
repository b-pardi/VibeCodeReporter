I will add more details and format later when it's not 2am, for now just semi jumbled notes on the repo structure and what ive done

So we can now take a given github url or urls from a text file (`data/repo_links.txt`), and grab all commits and all information from those commits, including the diff file of the commit. These are outputted as csvs, however sqlite may be a better alternative down the line is pandas is too slow. 

WIP script to web scrape some github projects to give to pydriller, and am working on being able to identify the domain of CS the project pertains to.

main.py is currently empty, I figure it will serve to tie all facets of the project together at a later point. For now the individual project components have entrypoints themselves (see `src/`).
src scripts will be the main components of the project, containing code that is capable of being run itself. Scripts that are smaller snippits that would otherwise clutter up code in the src scripts and/or be useful in other scripts as well, go in `utils/`.
configs: contain all dataclasses for user configurable parameters and things that will need to frequently and manually change as the project grows
