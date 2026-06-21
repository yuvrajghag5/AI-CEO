
import json
import os
import requests
import re
import html
from config.paths import RAW_DIR
from config.settings import NOD
 
QUERY =  "NVIDIA"           # <-- company/keyword to search
MAX_RESULTS = NOD
MAX_COMMENTS_PER_STORY = 15
OUTPUT_FILE = RAW_DIR / "hackernews_data.json"
SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ITEM_URL = "https://hn.algolia.com/api/v1/items/{}"
 
 
 
def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
 
def fetch_comments(object_id, max_comments=MAX_COMMENTS_PER_STORY):
    """
    Fetch the full discussion thread for a story and pull out comment
    text (this is where the actual content lives — most HN stories
    are bare links with no body text of their own).
    """
    try:
        response = requests.get(ITEM_URL.format(object_id), timeout=15)
        response.raise_for_status()
        item = response.json()
    except requests.RequestException as e:
        print(f"  failed to fetch comments for item {object_id}: {e}")
        return ""
 
    texts = []
 
    def walk(node):
        if len(texts) >= max_comments:
            return
        for child in node.get("children", []):
            text = child.get("text")
            if text:
                texts.append(text)
            walk(child)
 
    walk(item)
    # strip basic HTML tags HN comments commonly contain, then decode entities
    cleaned = [html.unescape(re.sub(r"<[^>]+>", " ", t)) for t in texts[:max_comments]]
    return "\n\n".join(cleaned)
 
 
def fetch_stories():
    params = {
        "query": QUERY,
        "tags": "story",
        "hitsPerPage": MAX_RESULTS,
    }
    response = requests.get(SEARCH_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
 
    stories = []
    for hit in data.get("hits", []):
        title = hit.get("title") or ""
        story_text = hit.get("story_text") or ""
        object_id = hit.get("objectID")
        hn_url = f"https://news.ycombinator.com/item?id={object_id}"
 
        print(f"Fetching comments for: {title}")
        comment_text = fetch_comments(object_id)
 
        # prefer comments (the real discussion), then story body, then title
        content = comment_text or story_text or title
 
        stories.append({
            "source": "hackernews",
            "title": title,
            "content": content,
            "url": hn_url,
            "published_date": hit.get("created_at"),
        })
    return stories
 
 
def main():
    existing = load_existing(OUTPUT_FILE)
    existing_urls = {item["url"] for item in existing}
 
    new_stories = fetch_stories()
    added = 0
    for story in new_stories:
        if story["url"] and story["url"] not in existing_urls:
            existing.append(story)
            existing_urls.add(story["url"])
            added += 1
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
 
    print(f"Fetched {len(new_stories)} stories, added {added} new, total {len(existing)} saved to {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()
 