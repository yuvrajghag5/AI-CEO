import json
from datetime import datetime, timezone
import chromadb
from config.paths import VECTORDB, EVIDENCE
from config.settings import TOP_K_PER_ANCHOR, CANDIDATE_POOL_SIZE

CHROMA_DIR = VECTORDB / "chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
OUTPUT_FILE = EVIDENCE / "evidence.json"

RELEVANT_COMPANIES = [
    "nvidia",
    "amd",
    "intel",
    "qualcomm",
    "broadcom",
    "tsmc",
    "samsung",
    "arm",
    "micron",
    "asml",
]

ANCHOR_PHRASES = {
    "opportunities": [
        "new product launch",
        "partnership announcement",
        "emerging technology adoption",
        "new market expansion",
    ],
    "risks": [
        "regulatory investigation",
        "competitive threat",
        "supply chain disruption",
        "negative public sentiment",
    ],
    "trends": [
        "industry shift",
        "customer behavior change",
        "technology adoption trend",
    ],
    "competitor_activity": [
        "competitor product launch",
        "competitor partnership",
        "competitor market share",
    ],
}

SENTIMENT_WEIGHT = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def recency_weight(published_date_str):
    """More recent documents get a higher weight (0 to 1).
    Falls back to a neutral 0.5 if the date can't be parsed."""
    if not published_date_str:
        return 0.5
    try:
        published = datetime.fromisoformat(published_date_str.replace("Z", "+00:00"))
        days_old = (datetime.now(timezone.utc) - published).days
        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return 0.75
        elif days_old <= 90:
            return 0.5
        else:
            return 0.25
    except (ValueError, TypeError):
        return 0.5


def compute_confidence(matches, category):
    """
    Rule-based confidence score (0 to 1), based on:
      - number of unique source documents matching the anchor
      - average sentiment alignment (risks favor negative sentiment,
        opportunities/trends favor positive/neutral)
      - average recency of the matching documents
    """
    unique_doc_ids = {m["doc_id"] for m in matches}
    source_count_score = min(len(unique_doc_ids) / 5, 1.0)  # caps out at 5+ sources

    if category == "risks":
        sentiment_scores = [
            1.0 - SENTIMENT_WEIGHT.get(m["sentiment_label"], 0.5) for m in matches
        ]
    elif category == "competitor_activity":
        # sentiment direction isn't meaningful here — competitor news being
        # "positive" isn't a good or bad sign for us either way
        sentiment_scores = [0.5 for _ in matches]
    else:
        sentiment_scores = [
            SENTIMENT_WEIGHT.get(m["sentiment_label"], 0.5) for m in matches
        ]
    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5

    recency_scores = [recency_weight(m["published_date"]) for m in matches]
    avg_recency_score = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5

    confidence = round(
        0.5 * source_count_score + 0.3 * avg_sentiment_score + 0.2 * avg_recency_score, 2
    )
    return confidence


def impact_tier(confidence):
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    return "Low"


def is_relevant(title, excerpt):
    """True if the chunk mentions NVIDIA or a known competitor."""
    text = f"{title or ''} {excerpt or ''}".lower()
    return any(company in text for company in RELEVANT_COMPANIES)


def search_anchor(collection, phrase):
    results = collection.query(query_texts=[phrase], n_results=CANDIDATE_POOL_SIZE)

    matches = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        title = meta.get("title")
        excerpt = results["documents"][0][i]

        if not is_relevant(title, excerpt):
            continue

        matches.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "title": title,
            "url": meta.get("url"),
            "published_date": meta.get("published_date"),
            "sentiment_label": meta.get("sentiment_label", "neutral"),
            "excerpt": excerpt,
        })

        if len(matches) >= TOP_K_PER_ANCHOR:
            break

    return matches


def search_topic_in_category(collection, category, topic):
    """
    Retrieve chunks that are about the USER'S topic AND carry the flavor
    of this category. We embed the topic fused with each of the category's
    anchor phrases, so e.g. "supply chain" under "risks" searches things
    like "supply chain disruption", "supply chain regulatory investigation",
    etc. — not bare "supply chain", which would return the same chunks for
    every category.

    Returns a deduped, recency-capped list of matches.
    """
    seen_chunks = {}
    for anchor in ANCHOR_PHRASES[category]:
        fused_query = f"{topic} {anchor}"          # topic CROSSED with category lens
        for m in search_anchor(collection, fused_query):
            seen_chunks[m["chunk_id"]] = m          # dedupe across anchors

    matches = list(seen_chunks.values())

    # We looped over several anchors, so cap the combined pool. Prefer the
    # most recent matches when trimming.
    matches.sort(key=lambda m: recency_weight(m["published_date"]), reverse=True)
    return matches[:CANDIDATE_POOL_SIZE]


def build_category_items(collection, category, phrases):
    items = []
    for phrase in phrases:
        matches = search_anchor(collection, phrase)
        if not matches:
            continue

        confidence = compute_confidence(matches, category)
        unique_sources = list({m["doc_id"] for m in matches})

        items.append({
            "title": phrase,
            "confidence": confidence,
            "impact": impact_tier(confidence),
            "supporting_doc_ids": unique_sources,
            "evidence": [
                {
                    "doc_id": m["doc_id"],
                    "source": m["source"],
                    "title": m["title"],
                    "url": m["url"],
                    "excerpt": m["excerpt"],
                }
                for m in matches[:3]  # keep only top 3 excerpts per item for readability
            ],
        })
    return items


def build_category_items_for_topic(collection, category, topic):
    """
    Like build_category_items, but for a single free-text topic supplied
    at question-time by the agent (instead of the fixed ANCHOR_PHRASES
    list). Used by tools.py so a question like "Should I invest in supply
    chain or manufacturing?" gets evidence that is BOTH about that topic
    AND shaped by the category lens (risk / opportunity / trend), rather
    than the same topic-only chunks for every category.
    """
    matches = search_topic_in_category(collection, category, topic)
    if not matches:
        return []

    confidence = compute_confidence(matches, category)
    unique_sources = list({m["doc_id"] for m in matches})

    return [{
        "title": topic,
        "confidence": confidence,
        "impact": impact_tier(confidence),
        "supporting_doc_ids": unique_sources,
        "evidence": [
            {
                "doc_id": m["doc_id"],
                "source": m["source"],
                "title": m["title"],
                "url": m["url"],
                "excerpt": m["excerpt"],
            }
            for m in matches[:3]
        ],
    }]


def main():
    collection = get_collection()
    if collection.count() == 0:
        print("ChromaDB collection is empty. Run store.py first.")
        return

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "opportunities": build_category_items(collection, "opportunities", ANCHOR_PHRASES["opportunities"]),
        "risks": build_category_items(collection, "risks", ANCHOR_PHRASES["risks"]),
        "trends": build_category_items(collection, "trends", ANCHOR_PHRASES["trends"]),
        "competitor_activity": build_category_items(collection, "competitor_activity", ANCHOR_PHRASES["competitor_activity"]),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    print(f"Opportunities found: {len(evidence['opportunities'])}")
    print(f"Risks found: {len(evidence['risks'])}")
    print(f"Trends found: {len(evidence['trends'])}")
    print(f"Competitor activity items found: {len(evidence['competitor_activity'])}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()