
from langchain_core.tools import tool
from engine.engine import (
    get_collection,
    build_category_items,
    build_category_items_for_topic,
    ANCHOR_PHRASES,
)
 
 
def _format_items(items):
    if not items:
        return "No relevant items found in the knowledge base for this category."
 
    lines = []
    for item in items:
        lines.append(f"- {item['title']} (confidence: {item['confidence']}, impact: {item['impact']})")
        for ev in item.get("evidence", [])[:2]:
            lines.append(f"    [{ev['source']}] {ev['title']} — {ev['excerpt'][:200]}")
    return "\n".join(lines)
 
 
@tool
def get_opportunities(topic: str = "") -> str:
    """Find strategic opportunities for NVIDIA. Use when the user asks
    about growth, new markets, partnerships, product launches, or upside
    scenarios. If topic is left empty, returns the general opportunity
    scan; if a topic is given, searches specifically for that wording."""
    collection = get_collection()
    if topic.strip():
        items = build_category_items_for_topic(collection, "opportunities", topic)
    else:
        items = build_category_items(collection, "opportunities", ANCHOR_PHRASES["opportunities"])
    return _format_items(items)
 
 
@tool
def get_risks(topic: str = "") -> str:
    """Find strategic risks facing NVIDIA. Use when the user asks about
    threats, regulatory issues, competition, supply chain problems, or
    downside scenarios. If topic is left empty, returns the general risk
    scan; if a topic is given, searches specifically for that wording."""
    collection = get_collection()
    if topic.strip():
        items = build_category_items_for_topic(collection, "risks", topic)
    else:
        items = build_category_items(collection, "risks", ANCHOR_PHRASES["risks"])
    return _format_items(items)
 
 
@tool
def get_trends(topic: str = "") -> str:
    """Find emerging industry or technology trends relevant to NVIDIA.
    Use when the user asks about industry shifts, customer behavior
    changes, or technology adoption trends."""
    collection = get_collection()
    if topic.strip():
        items = build_category_items_for_topic(collection, "trends", topic)
    else:
        items = build_category_items(collection, "trends", ANCHOR_PHRASES["trends"])
    return _format_items(items)
 
 
@tool
def get_competitor_activity(topic: str = "") -> str:
    """Find recent competitor moves (AMD, Intel, Qualcomm, etc.) relevant
    to NVIDIA. Use when the user asks what competitors are doing, market
    share shifts, or competitive positioning."""
    collection = get_collection()
    if topic.strip():
        items = build_category_items_for_topic(collection, "competitor_activity", topic)
    else:
        items = build_category_items(collection, "competitor_activity", ANCHOR_PHRASES["competitor_activity"])
    return _format_items(items)
 
 
@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Free-text search over all collected NVIDIA documents (news, RSS,
    Hacker News). Use this for specific factual questions that don't
    clearly fall into opportunities, risks, trends, or competitor
    activity — e.g. "What did NVIDIA announce at GTC?" """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
 
    if not results["ids"][0]:
        return "No matching documents found."
 
    lines = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        lines.append(f"[{meta.get('source')}] {meta.get('title')} — {text[:200]}")
    return "\n".join(lines)
 
 
ALL_TOOLS = [get_opportunities, get_risks, get_trends, get_competitor_activity, search_knowledge_base]
 