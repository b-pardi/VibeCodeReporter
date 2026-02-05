import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


url = "https://api.github.com/search/repositories"
current_dir = Path(__file__).parent 
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')


def search_github_by_topic(topic, filters=[], limit=500, language="python"):
    
    all_repos = []
    
    for page in range(1, (limit // 100) + 1):
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
                
        except requests.exceptions.HTTPError as err:
            print(f"HTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

    return all_repos

def save_data(data, path=current_dir):
    print(f"Saving {len(data['items'])} repositories to {path}")

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def mine_github_repos():
    # Define topics, languages, minimum stars, and limits
    topics = [
        ("machine-learning", "python", 50, 100),
        ("machine-learning", "c++", 50, 100),
        ("machine-learning", "java", 50, 100),
        ("frontend", "javascript", 50, 200),
        ("frontend", "python", 50, 100),
        ("frontend", "java", 50, 100),
        ("database", "python", 50, 100),
        ("database", "java", 50, 100),
        ("database", "javascript", 50, 100),
        ("cybersecurity", "python", 50, 250),
        ("cybersecurity", "c", 50, 100),
        ("cybersecurity", "go", 50, 100),
        ("cybersecurity", "c++", 50, 100),
        ("cybersecurity", "java", 50, 100),
        ("gamedev", "c++", 50, 100),
        ("gamedev", "c#", 50, 100),
        ("gamedev", "javascript", 50, 100),
        ("gamedev", "python", 50, 100),
        ("gamedev", "java", 50, 100),
        ("compiler", "c", 50, 100),
        ("compiler", "c++", 50, 100),
        ("compiler", "python", 50, 100)
    ]
    
    for topic, language, min_stars, limit in topics:
        path = current_dir.joinpath('..', 'data', 'github_repos', 'pre2022', f'{topic}_{language}_repos.json')

        #This filters help filter out repositories that aren't projects
        filters = [
            f"topic:{topic}",
            f"stars:>{min_stars}",
            f"language:{language}",
            f"created:<2022-01-01",
        ]
        repos = search_github_by_topic(topic, filters, limit=limit, language=language)
        save_data({"items": repos}, path=path)

    for topic, language, min_stars, limit in topics:
        path = current_dir.joinpath('..', 'data', 'github_repos','post2022', f'{topic}_{language}_repos.json')

        #This filters help filter out repositories that aren't projects
        filters = [
            f"topic:{topic}",
            f"stars:>{min_stars}",
            f"language:{language}",
            f"created:>2023-01-01",
        ]
        repos = search_github_by_topic(topic, filters, limit=limit, language=language)
        save_data({"items": repos}, path=path)

if __name__ == "__main__":
    mine_github_repos()
