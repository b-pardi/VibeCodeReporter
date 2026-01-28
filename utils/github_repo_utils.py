import requests
import json
from pathlib import Path

url = "https://api.github.com/search/repositories"
current_dir = Path(__file__).parent 
GITHUB_TOKEN = "YOUR_TOKEN" #TODO: Replace with env var


def search_github_by_topic(topic, min_stars=100, limit=500, language="python"):
    
    all_repos = []
    
    for page in range(1, (limit // 100) + 1):
        filters = [ f"topic:{topic}", 
                "has:license", 
                f"stars:>{min_stars}", 
                f"language:{language}"
        ]
        query = " ".join(filters)

        print(f"Fetching page {page}... for topic '{topic}'")
        
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": 1,
            "private": "false"
        }
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_TOKEN}"
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status() 
            
            data = response.json()
            all_repos.extend(data.get('items', []))

            print(f"Total Results Found: {data['total_count']}\n")
                
        except requests.exceptions.HTTPError as err:
            print(f"HTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

    save_data({"items": all_repos}, name=f'github_repos_{topic}.json')

def save_data(data, name = 'github_repos.json'):
    path = current_dir.joinpath('..', 'data', 'github_repos')
    print(f"Saving {len(data['items'])} repositories to {path.joinpath(name)}")

    with open(path.joinpath(name), 'w') as f:
        json.dump(data, f, indent=4)

search_github_by_topic("machine-learning")
search_github_by_topic("web-development", language="javascript")
search_github_by_topic("mobile-development", language="java")
search_github_by_topic("gamedev", language="c++")