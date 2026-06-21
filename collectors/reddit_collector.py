"""
Reddit collector.
Fetches posts from subreddits via Reddit's public JSON endpoint and
appends new ones to reddit_data.json (skips duplicates already saved,
based on URL).
"""
import json
import os
import requests
from config.settings import NOD
from config.paths import RAW_DIR

SUBREDDITS = ["nvidia", "hardware", "investing", "AI", "DATACENTERS", "GPU", "Jensen_Huang"]   # <-- edit as needed
LIMIT = NOD
OUTPUT_FILE = RAW_DIR / "reddit_data.json"
HEADERS = {"User-Agent": "ai-ceo-strategic-intelligence-agent/1.0"}


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def fetch_subreddit(subreddit):
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"limit": LIMIT, "t": "month"}
    response = requests.get(url, headers=HEADERS, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        posts.append({
            "source": "reddit",
            "title": post.get("title", ""),
            "content": post.get("selftext") or post.get("title", ""),
            "url": f"https://reddit.com{post.get('permalink', '')}",
            "published_date": post.get("created_utc"),
        })
    return posts


def main():
    existing = load_existing(OUTPUT_FILE)
    existing_urls = {item["url"] for item in existing}

    all_new_posts = []
    for subreddit in SUBREDDITS:
        try:
            all_new_posts.extend(fetch_subreddit(subreddit))
        except requests.RequestException as e:
            print(f"Failed to fetch r/{subreddit}: {e}")

    added = 0
    for post in all_new_posts:
        if post["url"] and post["url"] not in existing_urls:
            existing.append(post)
            existing_urls.add(post["url"])
            added += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Fetched {len(all_new_posts)} posts, added {added} new, total {len(existing)} saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
