"""
RAG retrieval for the CEO briefing  (Box 5 -- retrieval half)

ONE job: given a topic, run seek() across all four categories
(risks/opportunities/trends/competitor_activity), flatten the results
into one numbered evidence block (S1, S2, ...) plus a source lookup.
This is the "retrieve + augment" half of RAG -- embed the topic,
pull matching chunks from ChromaDB via seek(), assemble them for the
prompt. The "generate" half lives in agent/briefing.py, which takes
this evidence and runs schema-constrained LLM generation on top of it.

Deliberately self-contained: re-runs seek() itself for all four
categories rather than depending on what any caller already gathered.

Place at: rag/rag.py
(rag/ needs an __init__.py -- can be empty -- for this import to work:
 from rag.rag import gather_evidence)
"""
from engine.engine import seek

CATEGORIES = ["risks", "opportunities", "trends", "competitor_activity"]


def gather_evidence(topic):
    """
    Run seek() for all four categories, flatten into one numbered
    evidence block (S1, S2, ...) -- the only valid citation IDs.
    Returns (formatted_text, source_lookup).
    """
    lines = []
    source_lookup = {}
    sid = 1

    for category in CATEGORIES:
        result = seek(category, topic)
        if not result["evidence"]:
            continue
        lines.append(f"\n{category.upper().replace('_', ' ')} "
                     f"(confidence: {result['confidence']}, impact: {result['impact']}):")
        for ev in result["evidence"]:
            source_id = f"S{sid}"
            lines.append(f"  [{source_id}] [{ev['source']}] {ev['title']} — {ev['excerpt'][:200]}")
            source_lookup[source_id] = ev
            sid += 1

    formatted = "\n".join(lines) if lines else "(no relevant evidence found)"
    return formatted, source_lookup