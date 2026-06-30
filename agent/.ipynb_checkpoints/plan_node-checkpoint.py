"""
Plan node (Box 4 — GOAL -> PLAN stage).

A separate, tools-free generation call: the model states its intent
(which lenses look relevant, whether a full briefing is warranted)
BEFORE any tool-calling happens.

Output goes into state["plan"] for DISPLAY ONLY and is NEVER written
back into state["messages"], so it cannot influence the tool-calling
prompt reason_node builds next.

Merges the plan instruction INTO the last user message (on a shallow
copy) instead of appending a second user turn -- Mistral's chat template
rejects two consecutive user messages.

PATCH: capped at 80 new tokens AND truncated at the first blank line or
numbered-list line. Previously the model wrote its 1-sentence plan then
kept going, inventing ungrounded facts (specific product names, dollar
figures) BEFORE any retrieval -- exactly the hallucination this whole
architecture exists to prevent. Harmless (plan is a side-channel, never
fed back) but it looks bad in a demo. Now it's clipped to the actual
plan statement.

Place at: agent/plan_node.py
"""
import re

from agent.model import get_base

PLAN_MAX_NEW_TOKENS = 80
RAW_DEBUG = False  # set True to print the raw plan during an oral-exam walkthrough

_PLAN_INSTRUCTION = (
    "\n\nBefore answering, in 1-2 short sentences, state your plan: "
    "which of these look relevant to answering my question -- "
    "checking risks, opportunities, trends, competitor activity, "
    "or generating a full CEO briefing -- and why."
)

# cut the plan at the first blank line OR the first numbered-list line --
# everything after that is the model drifting into pre-retrieval facts.
_PLAN_CLIP = re.compile(r"\n\s*\n|\n\s*\d+\.")


def plan_node(state: dict) -> dict:
    tokenizer, model = get_base()

    plan_messages = [dict(m) for m in state["messages"]]
    plan_messages[-1]["content"] = plan_messages[-1]["content"] + _PLAN_INSTRUCTION

    # no tools= argument -- plain text generation, cannot trigger [TOOL_CALLS].
    prompt = tokenizer.apply_chat_template(
        plan_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=PLAN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    plan_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # clip to the plan statement only
    plan_text = _PLAN_CLIP.split(plan_text)[0].strip()

    if RAW_DEBUG:
        print("\n----- PLAN -----")
        print(plan_text)
        print("----- end plan -----\n")

    return {"plan": plan_text}