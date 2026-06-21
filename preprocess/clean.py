
import json
import os
import re
import html
import hashlib
import uuid
from config.paths import CLEAN_DIR, RAW_DIR 



INPUT_FILES = {
    "newsapi": RAW_DIR / "newsapi_data.json",
    "rss": RAW_DIR / "rss_data.json",
    "hackernews": RAW_DIR / "hackernews_data.json",
}
OUTPUT_FILE = CLEAN_DIR / "clean_documents.json"
MIN_CONTENT_LENGTH = 20   
 
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  warning: {path} not found, skipping")
    return []
 
 
def basic_clean(text):
    """Shared cleanup applied to every source: decode entities, strip
    HTML tags (in case any slipped through), normalize whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
 
 
def clean_newsapi(doc):
    doc["title"] = basic_clean(doc.get("title", ""))
    doc["content"] = basic_clean(doc.get("content", ""))
    return doc
 
 
def clean_rss(doc):
    doc["title"] = basic_clean(doc.get("title", ""))
    doc["content"] = basic_clean(doc.get("content", ""))
    return doc
 
 
def clean_hackernews(doc):
    doc["title"] = basic_clean(doc.get("title", ""))
    content = basic_clean(doc.get("content", ""))
    # strip quoted reply markers (e.g. "> some quoted text") left over from comments
    content = re.sub(r"(^|\s)>\s*", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    doc["content"] = content
    return doc
 
 
CLEANERS = {
    "newsapi": clean_newsapi,
    "rss": clean_rss,
    "hackernews": clean_hackernews,
}
 
 
def dedup_key(doc):
    """Plain normalized URL (or title if URL missing) used to detect
    duplicates. Not hashed — easier to debug, and we don't need it to
    look like an ID since doc_id is now a separate simple integer."""
    return (doc.get("url") or doc.get("title") or "").strip().lower()
 
 
def is_valid(doc):
    if not doc.get("title") and not doc.get("content"):
        return False
    if len(doc.get("content", "")) < MIN_CONTENT_LENGTH:
        return False
    return True
 
 
def main():
    # Step 1: load existing clean_documents.json if present, else start fresh
    existing_docs = load_json(OUTPUT_FILE)
    if existing_docs:
        print(f"Found existing {OUTPUT_FILE} with {len(existing_docs)} documents — will append on top of it.")
    else:
        print(f"No existing {OUTPUT_FILE} found — creating it from scratch.")
 
    existing_keys = {dedup_key(doc) for doc in existing_docs}
    max_existing_id = max((doc.get("doc_id", 0) for doc in existing_docs), default=0)
 
    # Step 2: clean each raw source file fresh
    new_candidates = []
    for source, filename in INPUT_FILES.items():
        raw_docs = load_json(filename)
        cleaner = CLEANERS[source]
 
        kept = 0
        for doc in raw_docs:
            doc = cleaner(doc)
            if not is_valid(doc):
                continue
            new_candidates.append(doc)
            kept += 1
 
        print(f"{source}: {len(raw_docs)} loaded -> {kept} valid after cleaning")
 
    # Step 3: drop anything that duplicates an existing doc OR a doc
    # already seen earlier in this same run (cross-source duplicates)
    seen_this_run = set()
    surviving_new_docs = []
    for doc in new_candidates:
        key = dedup_key(doc)
        if key in existing_keys or key in seen_this_run:
            continue
        seen_this_run.add(key)
        surviving_new_docs.append(doc)
 
    skipped = len(new_candidates) - len(surviving_new_docs)
    print(f"\n{len(new_candidates)} candidate documents -> {skipped} duplicates skipped -> {len(surviving_new_docs)} genuinely new")
 
    # Step 4: assign incrementing integer doc_ids continuing from the highest existing one
    next_id = max_existing_id + 1
    for doc in surviving_new_docs:
        doc["doc_id"] = next_id
        next_id += 1
 
    # Step 5: append and save
    combined = existing_docs + surviving_new_docs
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
 
    print(f"Added {len(surviving_new_docs)} new documents (doc_id {max_existing_id + 1} to {next_id - 1}).")
    print(f"Total documents in {OUTPUT_FILE}: {len(combined)}")
 
 
if __name__ == "__main__":
    main()