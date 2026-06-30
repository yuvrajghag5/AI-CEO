"""
CEO Briefing generation  (Box 5 -- generation half)

Takes evidence already gathered by rag/rag.py (the retrieval half) and
runs schema-constrained generation against it via outlines, producing a
validated CEOBriefing dict. The retrieval/RAG logic lives in rag/rag.py;
the prompt text lives in agent/prompt.py -- this file is just the
generation step that ties them together.

Place at: agent/briefing.py
Run standalone: python -m agent.briefing
"""
import json

import outlines

from config.settings import MAX_NEW_TOKENS_AGENT
from agent.model import get_outlines_model
from agent.schema import CEOBriefing
from agent.prompt import PROMPT_TEMPLATE
from rag.rag import gather_evidence

COMPANY = "NVIDIA"


def generate_ceo_briefing_dict(topic: str) -> dict:
    """
    Produce a schema-valid CEO briefing dict for the given topic.
    Returns {"error": ...} instead of raising if there's no evidence at
    all, so a calling tool gets a usable observation rather than a crash.
    """
    evidence_text, source_lookup = gather_evidence(topic)
    if not source_lookup:
        return {
            "error": (
                f"No internal evidence was found for the topic '{topic}' "
                f"across risks, opportunities, trends, or competitor "
                f"activity. Do NOT answer using outside or pretrained "
                f"knowledge. Tell the user plainly that no internal "
                f"evidence exists for this specific topic, and suggest a "
                f"more specific or differently-worded topic (for example "
                f"naming a company, product, or event)."
            )
        }

    model = get_outlines_model()  # shared weights -- no second model load
    prompt_text = PROMPT_TEMPLATE.format(company=COMPANY, question=topic, evidence=evidence_text)

    generator = outlines.Generator(model, CEOBriefing)
    result = generator(prompt_text, max_new_tokens=MAX_NEW_TOKENS_AGENT)

    # outlines versions differ: already-parsed object vs raw JSON string
    if isinstance(result, CEOBriefing):
        briefing_obj = result
    else:
        briefing_obj = CEOBriefing.model_validate_json(result)

    briefing_dict = briefing_obj.model_dump()
    briefing_dict["_sources"] = source_lookup  # real citations for the dashboard
    return briefing_dict


if __name__ == "__main__":
    topic = input("Enter a topic for the CEO briefing (e.g. 'AI infrastructure partnerships'): ").strip()
    if not topic:
        topic = "AI infrastructure partnerships"
        print(f"(empty — using '{topic}')")

    print("\nGenerating CEO briefing... this can take a while (schema-constrained generation).\n")
    briefing = generate_ceo_briefing_dict(topic)
    print(json.dumps(briefing, indent=2, ensure_ascii=False))