"""
Validate node (Box 4 — VALIDATE stage).

Pure deterministic Python -- zero LLM calls, so it can't introduce any
new tool-calling risk. Two checks:

  1. If a full briefing was generated, every supporting_evidence ID
     (S1, S2, ...) referenced anywhere in it must actually exist in its
     own _sources lookup. Catches invented citation IDs.
  2. Any http(s) URL in the final free-text answer must appear verbatim
     somewhere in the evidence actually gathered this turn. Catches
     fabricated links (the fake newsapi.org URL bug observed earlier).

Does not rewrite or censor the answer -- just attaches a validation
report so the result is honest about what was actually verified.

Place at: agent/validate_node.py
"""
import json
import re

RAW_DEBUG = True

URL_PATTERN = re.compile(r"https?://[^\s)\]]+")


def validate_node(state: dict) -> dict:
    evidence_list = state["evidence"]
    messages = state["messages"]

    final_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_text = msg["content"]
            break

    flagged = []
    citations_checked = 0
    citations_verified = 0

    # --- check 1: briefing citation IDs ---
    briefing = None
    for ev in evidence_list:
        if ev.get("tool") == "generate_ceo_briefing":
            try:
                parsed = json.loads(ev["result"])
                if "error" not in parsed:
                    briefing = parsed
            except (json.JSONDecodeError, TypeError):
                pass

    if briefing is not None:
        valid_ids = set(briefing.get("_sources", {}).keys())
        sections = (
            briefing.get("key_opportunities", []) + briefing.get("key_risks", []) +
            briefing.get("competitor_activity", []) + briefing.get("emerging_trends", []) +
            briefing.get("strategic_recommendations", [])
        )
        for item in sections:
            for sid in item.get("supporting_evidence", []):
                citations_checked += 1
                if sid in valid_ids:
                    citations_verified += 1
                else:
                    flagged.append(f"citation '{sid}' does not match any real source")

    # --- check 2: fabricated URLs in the free-text answer ---
    gathered_text = "\n".join(
        ev.get("result", "") for ev in evidence_list if isinstance(ev.get("result"), str)
    )
    for url in URL_PATTERN.findall(final_text):
        citations_checked += 1
        if url in gathered_text:
            citations_verified += 1
        else:
            flagged.append(f"URL '{url}' not found in gathered evidence (possibly fabricated)")

    validation = {
        "citations_checked": citations_checked,
        "citations_verified": citations_verified,
        "flagged": flagged,
        "passed": len(flagged) == 0,
    }

    if RAW_DEBUG:
        print("\n----- VALIDATE -----")
        print(json.dumps(validation, indent=2))
        print("----- end validate -----\n")

    return {"validation": validation}