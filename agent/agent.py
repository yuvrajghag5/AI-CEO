
import json
import os
import re
from datetime import datetime, timezone
import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.paths import VECTORDB, EVIDENCE
from config.settings import MODEL, TOP_K, MAX_NEW_TOKENS_AGENT, DO_SAMPLE, REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE, TOP_P, TEMPERATURE
from rag.prompt import CEO_PROMPT_TEMPLATE
 
EVIDENCE_FILE = EVIDENCE / "evidence.json"
CHROMA_DIR = VECTORDB / "./chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
OUTPUT_FILE = EVIDENCE / "ceo_report.json"
MODEL_NAME = MODEL   
COMPANY_NAME = "NVIDIA"
TOP_K_RETRIEVED_CONTEXT = TOP_K
MAX_NEW_TOKENS = MAX_NEW_TOKENS_AGENT
NO_REPEAT_NGRAM_SIZE = NO_REPEAT_NGRAM_SIZE
REPETITION_PENALTY = REPETITION_PENALTY
DO_SAMPLE = DO_SAMPLE
TOP_P = TOP_P
TEMPERATURE = TEMPERATURE
CEO_QUESTION = "If you were the CEO of NVIDIA today, what would you do next and why?"
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
 
 
def load_model():
    print(f"Loading {MODEL_NAME} ... (this can take a while on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model
 
 
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
    on top of the pre-computed evidence.json. Assigns [R1], [R2]... IDs.
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
    Basic post-processing for a local model's output:
      - cut off anything after the CEO Action Plan section (in case the
        model rambles on despite being told to stop)
      - warn (not crash) if expected sections are missing, so you know
        the run needs a retry rather than silently shipping a broken briefing
    """
    text = raw_output.strip()
 
    action_plan_match = re.search(r"7\.\s*CEO Action Plan", text, re.IGNORECASE)
    if action_plan_match:
        after_plan = text[action_plan_match.start():]
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
 
 
# Section headers exactly as required by CEO_PROMPT_TEMPLATE, in order.
SECTION_HEADERS = [
    ("executive_summary", r"1\.\s*Executive Summary"),
    ("key_opportunities", r"2\.\s*Key Opportunities"),
    ("key_risks", r"3\.\s*Key Risks"),
    ("competitor_activity", r"4\.\s*Competitor Activity"),
    ("emerging_trends", r"5\.\s*Emerging Trends"),
    ("strategic_recommendations", r"6\.\s*Strategic Recommendations"),
    ("ceo_action_plan", r"7\.\s*CEO Action Plan"),
]
 
 
def parse_briefing_sections(text):
    """
    Splits the model's raw structured output into a dict of named
    sections, so the dashboard can render each part separately
    (e.g. a Recommendations panel, an Action Plan panel) instead of
    only having one giant text blob.
    """
    positions = []
    for name, pattern in SECTION_HEADERS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            positions.append((name, match.start()))
 
    positions.sort(key=lambda p: p[1])
 
    sections = {}
    for i, (name, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        sections[name] = text[start:end].strip()
 
    # Make sure every expected key exists even if a section was missing
    for name, _ in SECTION_HEADERS:
        sections.setdefault(name, "")
 
    return sections
 
 
def generate_response(prompt_text, tokenizer, model):
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
 
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
 
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample= DO_SAMPLE,
        temperature = TEMPERATURE,
        top_p = TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=tokenizer.eos_token_id,
    )
 
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
 
 
def generate_briefing(question, tokenizer, model, collection, evidence):
    strategic_evidence = format_strategic_evidence(evidence)
    retrieved_context = format_retrieved_context(collection, question)
 
    prompt_text = CEO_PROMPT_TEMPLATE.format(
        company=COMPANY_NAME,
        question=question,
        strategic_evidence=strategic_evidence,
        retrieved_context=retrieved_context,
    )
 
    raw_output = generate_response(prompt_text, tokenizer, model)
    cleaned = validate_and_trim(raw_output)
    memory.append({"question": question, "answer": cleaned})
    return cleaned
 
 
def save_report(question, raw_briefing, sections, evidence_used, retrieved_context_used):
    report = {
        "company": COMPANY_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_question": question,
        "strategic_context_used": evidence_used,
        "retrieved_context_used": retrieved_context_used,
        "ceo_briefing_raw": raw_briefing,
        "sections": sections,
    }
 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
 
    print(f"\nSaved CEO report to {OUTPUT_FILE}")
 
 
def main():
    evidence = load_evidence()
    if evidence is None:
        return
 
    collection = get_collection()
    tokenizer, model = load_model()
 
    briefing = generate_briefing(CEO_QUESTION, tokenizer, model, collection, evidence)
    sections = parse_briefing_sections(briefing)
 
    print("\n" + "=" * 60)
    print(f"CEO QUESTION: {CEO_QUESTION}")
    print("=" * 60 + "\n")
    print(briefing)
 
    save_report(
        question=CEO_QUESTION,
        raw_briefing=briefing,
        sections=sections,
        evidence_used=True,
        retrieved_context_used=True,
    )
 
 
if __name__ == "__main__":
    main()
 