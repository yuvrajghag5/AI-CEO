"""
NewsAPI collector.
Fetches news articles and appends new ones to newsapi_data.json
(skips duplicates already saved, based on URL).
"""
import json
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from config.settings import NOD
from config.paths import RAW_DOCUMENTS_FILE
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")    
QUERY = '"NVIDIA" AND (earnings OR AI OR chip OR datacenter OR stock OR GPU OR Jensen Huang)'                     
OUTPUT_FILE = RAW_DOCUMENTS_FILE
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
    Returns "" if the page can't be fetched or has too little text
    (paywalls, JS-rendered pages, blocked bots are common and expected).
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
 
 
def fetch_article_metadata():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": NOD,
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
 
    if data.get("status") != "ok":
        print("NewsAPI error:", data.get("message"))
        return []
 
    return [
        {
            "source": "newsapi",
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "published_date": item.get("publishedAt"),
        }
        for item in data.get("articles", [])
    ]
 
 
def main():
    existing = load_existing(OUTPUT_FILE)
    existing_urls = {item["url"] for item in existing}
 
    candidates = fetch_article_metadata()
    added = 0
 
    for article in candidates:
        if not article["url"] or article["url"] in existing_urls:
            continue
 
        print(f"Scraping: {article['url']}")
        content = scrape_main_content(article["url"])
 
        if not content:
            print("  skipped (no usable content extracted)")
            continue
 
        article["content"] = content
        existing.append(article)
        existing_urls.add(article["url"])
        added += 1
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
 
    print(f"Found {len(candidates)} candidates, added {added} new, total {len(existing)} saved to {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()