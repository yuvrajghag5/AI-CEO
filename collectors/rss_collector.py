
import json
import os
import feedparser
import requests
from bs4 import BeautifulSoup

from config.paths import RAW_DIR
 
FEED_URLS = [
    "https://blogs.nvidia.com/feed/",
    "https://www.theverge.com/rss/index.xml",
]   # <-- edit as needed
MAX_ENTRIES_PER_FEED = 50
OUTPUT_FILE = RAW_DIR /"rss_data.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-ceo-collector/1.0)"}
MIN_CONTENT_LENGTH = 200
 
 
def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
 
def scrape_main_content(url):
    """
    Fetch the article page and extract its main text via <p> tags.
    Returns "" if the page can't be fetched or has too little text.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  scrape failed for {url}: {e}")
        return ""
 
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs)
 
    if len(text) < MIN_CONTENT_LENGTH:
        return ""
    return text
 
 
def fetch_feed_metadata(feed_url):
    parsed = feedparser.parse(feed_url)
 
    return [
        {
            "source": "rss",
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published_date": entry.get("published", None),
        }
        for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]
    ]
 
 
def main():
    existing = load_existing(OUTPUT_FILE)
    existing_urls = {item["url"] for item in existing}
 
    candidates = []
    for feed_url in FEED_URLS:
        candidates.extend(fetch_feed_metadata(feed_url))
 
    added = 0
    for entry in candidates:
        if not entry["url"] or entry["url"] in existing_urls:
            continue
 
        print(f"Scraping: {entry['url']}")
        content = scrape_main_content(entry["url"])
 
        if not content:
            print("  skipped (no usable content extracted)")
            continue
 
        entry["content"] = content
        existing.append(entry)
        existing_urls.add(entry["url"])
        added += 1
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
 
    print(f"Found {len(candidates)} candidates, added {added} new, total {len(existing)} saved to {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()