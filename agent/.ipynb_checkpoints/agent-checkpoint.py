

import json
import os
from datetime import datetime, timezone
import chromadb
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.paths import VECTORDB, EVIDENCE
from config.settings import MODEL, TOP_K, MAX_NEW_TOKENS_AGENT
from rag.prompt import CEO_PROMPT_TEMPLATE
from agent.schema import CEOBriefing
 
EVIDENCE_FILE = EVIDENCE / "evidence.json"
CHROMA_DIR = VECTORDB / "chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
OUTPUT_FILE = EVIDENCE / "ceo_report.json"
MODEL_NAME = MODEL
COMPANY_NAME = "NVIDIA"
TOP_K_RETRIEVED_CONTEXT = TOP_K
MAX_NEW_TOKENS = MAX_NEW_TOKENS_AGENT
 
CEO_QUESTION = "If you were the CEO of NVIDIA today, what would you do next and why?"
memory = []
 
 
def load_evidence():
    if not os.path.exists(EVIDENCE_FILE):
        print(f"{EVIDENCE_FILE} not found. Run engine.py first.")
        return None
    with open(EVIDENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)
 
 
def load_model():
    """
    Loads the model/tokenizer via transformers as usual, then wraps
    them with outlines so generation can be schema-constrained.
    """
    print(f"Loading {MODEL_NAME} ... (this can take a while on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model = outlines.from_transformers(hf_model, tokenizer)
    return tokenizer, model
 
 
def format_strategic_evidence(evidence):
    """
    Flatten evidence.json into plain text, assigning a source ID
    (e.g. S1, S2) to every individual evidence excerpt across all
    categories. These are the ONLY valid source IDs the LLM should
    cite — it should never invent its own.
 
    Returns (formatted_text, source_lookup) where source_lookup maps
    each ID to its actual evidence detail, so the dashboard can show
    the real text behind a citation like "S3" instead of just the ID.
    """
    lines = []
    source_lookup = {}
    source_id_counter = 1
 
    for category in ("opportunities", "risks", "trends", "competitor_activity"):
        items = evidence.get(category, [])
        if not items:
            continue
        lines.append(f"\n{category.upper().replace('_', ' ')}:")
 
        for item in items:
            lines.append(f"- {item['title']} (confidence: {item['confidence']}, impact: {item['impact']})")
            for ev in item.get("evidence", [])[:2]:
                source_id = f"S{source_id_counter}"
                lines.append(f"    [{source_id}] [{ev['source']}] {ev['title']} — {ev['excerpt'][:200]}")
                source_lookup[source_id] = {
                    "category": category,
                    "source": ev.get("source"),
                    "title": ev.get("title"),
                    "url": ev.get("url"),
                    "excerpt": ev.get("excerpt"),
                }
                source_id_counter += 1
 
    return "\n".join(lines), source_lookup
 
 
def format_retrieved_context(collection, question, top_k=TOP_K_RETRIEVED_CONTEXT):
    """
    Additional ChromaDB retrieval for the specific question being asked,
    on top of the pre-computed evidence.json. Assigns R1, R2... IDs.
 
    Returns (formatted_text, source_lookup), same purpose as
    format_strategic_evidence above.
    """
    results = collection.query(query_texts=[question], n_results=top_k, include=["metadatas", "documents", "distances"])
 
    lines = []
    source_lookup = {}
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        distance = results["distances"][0][i] if results.get("distances") else None
        retrieved_id = f"R{i + 1}"
        lines.append(f"[{retrieved_id}] [{meta.get('source')}] {meta.get('title')} — {text[:200]}")
        source_lookup[retrieved_id] = {
            "category": "retrieved_context",
            "source": meta.get("source"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "excerpt": text[:200],
            "score": round(1 - distance, 4) if distance is not None else None,
        }
 
    formatted = "\n".join(lines) if lines else "(no additional context retrieved)"
    return formatted, source_lookup
 
 
def build_sources_used(sources_lookup):
    """
    Groups the flat {source_id: detail} lookup by underlying document
    (title + url), merging every source_id that points to the same
    article into one entry — matching the "sources_used" format used
    elsewhere in the project (one row per real document, not per ID).
    """
    grouped = {}
 
    for source_id, detail in sources_lookup.items():
        key = (detail.get("title"), detail.get("url"))
        if key not in grouped:
            grouped[key] = {
                "source_ids": [],
                "categories": set(),
                "source": detail.get("source"),
                "title": detail.get("title"),
                "url": detail.get("url"),
                "score": detail.get("score"),
            }
        grouped[key]["source_ids"].append(source_id)
        grouped[key]["categories"].add(detail.get("category"))
 
    sources_used = []
    for entry in grouped.values():
        sources_used.append({
            "source_id": ", ".join(sorted(entry["source_ids"])),
            "type": "strategic_evidence" if any(sid.startswith("S") for sid in entry["source_ids"]) else "retrieved_context",
            "category": ", ".join(sorted(entry["categories"])),
            "title": entry["title"],
            "source": entry["source"],
            "url": entry["url"],
            "score": entry["score"],
        })
 
    return sources_used
 
 
def format_history(memory, max_turns=3):
    if not memory:
        return "(no previous conversation)"
    recent = memory[-max_turns:]
    return "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent)
 
 
def generate_briefing(question, model, collection, evidence):
    strategic_evidence, strategic_lookup = format_strategic_evidence(evidence)
    retrieved_context, retrieved_lookup = format_retrieved_context(collection, question)
    sources_lookup = {**strategic_lookup, **retrieved_lookup}
 
    prompt_text = CEO_PROMPT_TEMPLATE.format(
        company=COMPANY_NAME,
        question=question,
        strategic_evidence=strategic_evidence,
        retrieved_context=retrieved_context,
    )
 
    # outlines.Generator forces every generated token to keep the
    # output valid against CEOBriefing's schema. The result is already
    # a validated Pydantic object — no regex, no retry-on-malformed-JSON.
    generator = outlines.Generator(model, CEOBriefing)
    result = generator(prompt_text, max_new_tokens=MAX_NEW_TOKENS)
 
    # outlines versions differ on whether this returns an already-parsed
    # Pydantic object or a raw JSON string — handle both.
    if isinstance(result, CEOBriefing):
        briefing_obj = result
    else:
        briefing_obj = CEOBriefing.model_validate_json(result)
 
    briefing_dict = briefing_obj.model_dump()
    memory.append({"question": question, "answer": briefing_dict})
    return briefing_dict, sources_lookup
 
 
def save_report(question, briefing_dict, sources_lookup, evidence_used, retrieved_context_used):
    report = {
        "company": COMPANY_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_question": question,
        "strategic_context_used": evidence_used,
        "retrieved_context_used": retrieved_context_used,
    }
    report.update(briefing_dict)
    report["sources_used"] = build_sources_used(sources_lookup)  # one entry per real document
 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
 
    print(f"\nSaved CEO report to {OUTPUT_FILE}")
    return report
 
 
def main():
    evidence = load_evidence()
    if evidence is None:
        return
 
    collection = get_collection()
    tokenizer, model = load_model()
 
    briefing_dict, sources_lookup = generate_briefing(CEO_QUESTION, model, collection, evidence)
 
    report = save_report(
        question=CEO_QUESTION,
        briefing_dict=briefing_dict,
        sources_lookup=sources_lookup,
        evidence_used=True,
        retrieved_context_used=True,
    )
 
    print("\n" + "=" * 60)
    print(f"CEO QUESTION: {CEO_QUESTION}")
    print("=" * 60 + "\n")
    print(json.dumps(report, indent=2))
 
 
if __name__ == "__main__":
    main()