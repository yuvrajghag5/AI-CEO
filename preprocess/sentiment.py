
import json
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.paths import CLEAN_DIR
 
INPUT_FILE = CLEAN_DIR / "clean_documents.json"
OUTPUT_FILE = CLEAN_DIR / "sentiment_analysis.json"
 

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
 
 
def classify(compound_score):
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.03:
        return "negative"
    return "neutral"
 
 
def main():
    documents = load_json(INPUT_FILE)
    if not documents:
        print(f"No documents found in {INPUT_FILE}. Run clean.py first.")
        return
 
    existing_results = load_json(OUTPUT_FILE)
    scored_doc_ids = {entry["doc_id"] for entry in existing_results}
 
    if existing_results:
        print(f"Found existing {OUTPUT_FILE} with {len(existing_results)} scored documents.")
    else:
        print(f"No existing {OUTPUT_FILE} found — creating it from scratch.")
 
    to_score = [doc for doc in documents if doc.get("doc_id") not in scored_doc_ids]
 
    max_existing_id = max((entry.get("sentiment_id", 0) for entry in existing_results), default=0)
    next_id = max_existing_id + 1
 
    analyzer = SentimentIntensityAnalyzer()
    new_results = []
 
    for doc in to_score:
        text = f"{doc.get('title', '')}. {doc.get('content', '')}".strip()
        scores = analyzer.polarity_scores(text)
 
        new_results.append({
            "sentiment_id": next_id,
            "doc_id": doc.get("doc_id"),
            "source": doc.get("source"),
            "title": doc.get("title"),
            "content": doc.get("content"),
            "url": doc.get("url"),
            "published_date": doc.get("published_date"),
            "sentiment_label": classify(scores["compound"]),
            "sentiment_score": scores["compound"],
        })
        next_id += 1
 
    combined_results = existing_results + new_results
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
 
    print(f"Already scored (skipped): {len(scored_doc_ids)}")
    print(f"Newly scored: {len(new_results)} (sentiment_id {max_existing_id + 1} to {next_id - 1})")
    print(f"Total entries in {OUTPUT_FILE}: {len(combined_results)}")
 
 
if __name__ == "__main__":
    main()