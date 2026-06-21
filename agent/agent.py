
import json
import os
import re
import chromadb
from langchain_ollama import ChatOllama
from config.paths import VECTORDB, EVIDENCE
from config.paths import OLLAMA_MODEL, TOP_K
from rag.prompt import CEO_PROMPT_TEMPLATE
 
EVIDENCE_FILE = EVIDENCE / "evidence.json"
CHROMA_DIR = VECTORDB / "./chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
OLLAMA_MODEL = OLLAMA_MODEL   # <-- match whatever you pulled via `ollama pull`
COMPANY_NAME = "NVIDIA"
TOP_K_RETRIEVED_CONTEXT = TOP_K
 
# Simple in-session memory: list of {"question": ..., "answer": ...} dicts.
# Not persisted to disk — resets every time the script restarts.
# Significance: lets the agent handle follow-up questions (e.g. during
# the oral exam) with context from what was already discussed, instead
# of treating every question as if it's the first one ever asked.
memory = []
 
 
def load_evidence():
    if not os.path.exists(EVIDENCE_FILE):
        print(f"{EVIDENCE_FILE} not found. Run engine.py first.")
        return None
    with open(EVIDENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)
 
 
def format_strategic_evidence(evidence):
    """
    Flatten evidence.json into plain text, assigning a source ID
    (e.g. [S1], [S2]) to every individual evidence excerpt across all
    categories. These are the ONLY valid source IDs the LLM is allowed
    to cite — it should never invent its own.
    """
    lines = []
    source_id_counter = 1
 
    for category in ("opportunities", "risks", "trends", "competitor_activity"):
        items = evidence.get(category, [])
        if not items:
            continue
        lines.append(f"\n{category.upper().replace('_', ' ')}:")
 
        for item in items:
            lines.append(f"- {item['title']} (confidence: {item['confidence']}, impact: {item['impact']})")
            for ev in item.get("evidence", [])[:2]:
                source_id = f"[S{source_id_counter}]"
                lines.append(f"    {source_id} [{ev['source']}] {ev['title']} — {ev['excerpt'][:200]}")
                source_id_counter += 1
 
    return "\n".join(lines)
 
 
def format_retrieved_context(collection, question, top_k=TOP_K_RETRIEVED_CONTEXT):
    """
    Additional ChromaDB retrieval for the specific question being asked,
    on top of the pre-computed evidence.json. Assigns [R1], [R2]... IDs,
    continuing the numbering style but in a separate namespace from [S...].
    """
    results = collection.query(query_texts=[question], n_results=top_k)
 
    lines = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        retrieved_id = f"[R{i + 1}]"
        lines.append(f"{retrieved_id} [{meta.get('source')}] {meta.get('title')} — {text[:200]}")
 
    return "\n".join(lines) if lines else "(no additional context retrieved)"
 
 
def validate_and_trim(raw_output):
    """
    Basic post-processing for a small local model's output:
      - cut off anything after the CEO Action Plan section (in case the
        model rambles on despite being told to stop)
      - warn (not crash) if expected sections are missing, so you know
        the run needs a retry rather than silently shipping a broken briefing
    """
    text = raw_output.strip()
 
    # Trim anything after the last action item of the CEO Action Plan.
    action_plan_match = re.search(r"7\.\s*CEO Action Plan", text, re.IGNORECASE)
    if action_plan_match:
        after_plan = text[action_plan_match.start():]
        # keep up through item "3." of the action plan, drop anything further
        trimmed = re.split(r"\n\s*(?=(?:Note|Disclaimer|---|\Z))", after_plan, maxsplit=1)[0]
        text = text[:action_plan_match.start()] + trimmed
 
    required_sections = [
        "Executive Summary",
        "Key Opportunities",
        "Key Risks",
        "Competitor Activity",
        "Emerging Trends",
        "Strategic Recommendations",
        "CEO Action Plan",
    ]
    missing = [s for s in required_sections if s.lower() not in text.lower()]
    if missing:
        print(f"  [warning] briefing is missing expected section(s): {missing} — consider regenerating")
 
    return text.strip()
 
 
def format_history(memory, max_turns=3):
    if not memory:
        return "(no previous conversation)"
    recent = memory[-max_turns:]
    return "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent)
 
 
def generate_briefing(question, llm, collection, evidence):
    strategic_evidence = format_strategic_evidence(evidence)
    retrieved_context = format_retrieved_context(collection, question)
 
    prompt_text = CEO_PROMPT_TEMPLATE.format(
        company=COMPANY_NAME,
        question=question,
        strategic_evidence=strategic_evidence,
        retrieved_context=retrieved_context,
    )
 
    response = llm.invoke(prompt_text)
    raw_output = response.content
 
    cleaned = validate_and_trim(raw_output)
    memory.append({"question": question, "answer": cleaned})
    return cleaned
 
 
def main():
    evidence = load_evidence()
    if evidence is None:
        return
 
    collection = get_collection()
    llm = ChatOllama(model=OLLAMA_MODEL)
 
    print("AI CEO Agent ready. Ask a CEO-style question, or type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
 
        briefing = generate_briefing(question, llm, collection, evidence)
        print(f"\n{briefing}\n")
 
 
if __name__ == "__main__":
    main()
 