"""
Strategic Intelligence Engine  (Box 2)

ONE job: given a category (opportunities / risks / trends /
competitor_activity) and the user's topic, return scored evidence for
that topic, pulled semantically from ChromaDB (filled by Box 1).

The four "seeker" tools in Box 3 each call seek() with a different
category. That's the whole connection between this file and the agent.

How seek() works, in plain steps:
  1. Take the user's topic (e.g. "supply chain").
  2. Fuse it with each of the category's anchor phrases, so under
     "risks" it searches "supply chain disruption", "supply chain
     regulatory investigation", etc. — risk-flavoured, not bare topic.
  3. Semantic-search ChromaDB for each fused phrase.
  4. Keep only chunks that actually mention NVIDIA or a competitor.
  5. Score how confident we are (more sources + right sentiment +
     more recent = higher), and return the top evidence.

Place at: engine/engine.py
Run standalone: python -m engine.engine
"""
from datetime import datetime, timezone

import chromadb

from config.paths import VECTORDB
from config.settings import TOP_K_PER_ANCHOR, CANDIDATE_POOL_SIZE

CHROMA_DIR = VECTORDB / "chroma_db"
COLLECTION_NAME = "ai_ceo_documents"

# a chunk only counts if it mentions one of these (keeps off-topic news out)
RELEVANT_COMPANIES = [
    "nvidia", "amd", "intel", "qualcomm", "broadcom",
    "tsmc", "samsung", "arm", "micron", "asml",
]

# the "flavour" phrases that turn a bare topic into a category-shaped search
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

_collection = None


def get_collection():
    """Open ChromaDB once and reuse it."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def recency_weight(published_date_str):
    """Newer documents score higher (1.0 down to 0.25). 0.5 if unknown."""
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


def is_relevant(title, excerpt):
    """True if this chunk actually mentions NVIDIA or a known competitor."""
    text = f"{title or ''} {excerpt or ''}".lower()
    return any(company in text for company in RELEVANT_COMPANIES)


def compute_confidence(matches, category):
    """
    A 0-1 score for how solid this evidence is:
      - 50%: how many distinct source documents back it (caps at 5)
      - 30%: sentiment fit (risks want negative news; opportunities/
             trends want positive; competitor news is neutral either way)
      - 20%: how recent the evidence is on average
    """
    unique_docs = {m["doc_id"] for m in matches}
    source_score = min(len(unique_docs) / 5, 1.0)

    if category == "risks":
        sentiment = [1.0 - SENTIMENT_WEIGHT.get(m["sentiment_label"], 0.5) for m in matches]
    elif category == "competitor_activity":
        sentiment = [0.5 for _ in matches]
    else:
        sentiment = [SENTIMENT_WEIGHT.get(m["sentiment_label"], 0.5) for m in matches]
    sentiment_score = sum(sentiment) / len(sentiment) if sentiment else 0.5

    recency = [recency_weight(m["published_date"]) for m in matches]
    recency_score = sum(recency) / len(recency) if recency else 0.5

    return round(0.5 * source_score + 0.3 * sentiment_score + 0.2 * recency_score, 2)


def impact_tier(confidence):
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    return "Low"


def _semantic_search(collection, phrase):
    """Embed `phrase`, pull nearest chunks, keep only NVIDIA-relevant ones."""
    results = collection.query(query_texts=[phrase], n_results=CANDIDATE_POOL_SIZE)

    matches = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        excerpt = results["documents"][0][i]
        title = meta.get("title")

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


def seek(category, topic):
    """
    THE function the tools call.

    category : "opportunities" | "risks" | "trends" | "competitor_activity"
    topic    : the user's question/topic, e.g. "supply chain"

    Returns a dict: category, topic, confidence, impact, and a list of
    the top evidence chunks. Empty evidence list if nothing relevant.
    """
    if category not in ANCHOR_PHRASES:
        raise ValueError(f"Unknown category: {category}. "
                         f"Expected one of {list(ANCHOR_PHRASES)}")

    collection = get_collection()

    # fuse topic with each anchor, dedupe chunks across anchors
    seen = {}
    for anchor in ANCHOR_PHRASES[category]:
        for m in _semantic_search(collection, f"{topic} {anchor}"):
            seen[m["chunk_id"]] = m

    matches = list(seen.values())
    if not matches:
        return {"category": category, "topic": topic,
                "confidence": 0.0, "impact": "Low", "evidence": []}

    # most recent first, then cap the pool
    matches.sort(key=lambda m: recency_weight(m["published_date"]), reverse=True)
    matches = matches[:CANDIDATE_POOL_SIZE]

    # confidence uses the FULL match pool (source_count_score already
    # counts unique doc_ids, so it's unaffected by what we do next)
    confidence = compute_confidence(matches, category)
    impact = impact_tier(confidence)

    # The evidence we SHOW, however, should come from distinct documents.
    # Without this, a topic that strongly matches only 1-2 articles ends
    # up returning 5 overlapping chunks of those same 1-2 articles --
    # looks like broad evidence, but it's really one source repeated.
    seen_docs = set()
    diverse_evidence = []
    for m in matches:
        doc_key = (m["source"], m["title"])
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        diverse_evidence.append(m)
        if len(diverse_evidence) >= 5:
            break

    return {
        "category": category,
        "topic": topic,
        "confidence": confidence,
        "impact": impact,
        "evidence": [
            {
                "source": m["source"],
                "title": m["title"],
                "url": m["url"],
                "excerpt": m["excerpt"],
            }
            for m in diverse_evidence
        ],
    }


if __name__ == "__main__":
    # Standalone smoke test — proves the engine identifies evidence
    # for a topic before any agent/tool exists.
    collection = get_collection()
    count = collection.count()
    print(f"ChromaDB has {count} chunks stored.\n")
    if count == 0:
        print("Collection is empty — run Box 1 (store.py) first.")
        raise SystemExit

    demos = [
        ("risks", "supply chain"),
        ("opportunities", "data center"),
        ("competitor_activity", "AMD"),
    ]
    for category, topic in demos:
        result = seek(category, topic)
        print("=" * 60)
        print(f"seek('{category}', '{topic}')")
        print(f"  confidence: {result['confidence']}  impact: {result['impact']}")
        print(f"  evidence pieces: {len(result['evidence'])}")
        for ev in result["evidence"][:3]:
            print(f"    [{ev['source']}] {ev['title']} — {ev['excerpt'][:120]}")
        print()