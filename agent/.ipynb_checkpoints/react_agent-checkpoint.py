"""
The Agent  (Box 4)  —  a real ReAct loop in LangGraph

WORKFLOW (Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate)
  Goal       : the user's question
  Plan       : plan_node (agent/plan_node.py) -- a tools-free intent statement
  Retrieve   : act_node / fallback_act -- executes the seeker tools
  Analyze    : seek()'s confidence scoring (engine/engine.py)
  Decide     : the model's synthesis in reason_node
  Recommend  : generate_ceo_briefing -> strategic_recommendations
  Validate   : validate_node (agent/validate_node.py) -- pure code citation check

  plan -> reason --(tool calls?)--> act --> reason --> ... --> validate --> END

Place at: agent/react_agent.py
Run: python -m agent.react_agent   (interactive: type questions, 'exit' to quit)
"""
import json
import re
import random
import string
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.model import get_base
from agent.tools import ALL_TOOLS
from agent.plan_node import plan_node
from agent.validate_node import validate_node

TOOL_SPECS = [convert_to_openai_tool(t) for t in ALL_TOOLS]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

REASON_MAX_NEW_TOKENS = 700

RAW_DEBUG = False  # set True to print raw model output (oral-exam tool-call demo)

# CONFIRMED BY A/B TEST: any system message suppresses Mistral's native
# [TOOL_CALLS] here. Routing guidance lives in the tool docstrings instead.
SYSTEM_PROMPT = None


# ----------------------------------------------------------------------
# robust parser: accept BOTH [TOOL_CALLS][{...}] and fenced/bare json
# ----------------------------------------------------------------------
def _extract_json_array(text):
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_tool_calls(text):
    """Return [{"name","arguments"}, ...] or None."""
    if "[TOOL_CALLS]" in text:
        arr = _extract_json_array(text.split("[TOOL_CALLS]", 1)[1])
        if arr:
            return [{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in arr]

    calls = []
    for block in re.findall(r"\{[^{}]*\"name\"[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if obj.get("name") in TOOLS_BY_NAME:
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
    return calls or None


def _gen_call_id():
    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


_RELEVANCE_KEYWORDS = [
    "nvidia", "amd", "intel", "qualcomm", "broadcom", "tsmc", "arm",
    "chip", "gpu", "semiconductor", "data center", "datacenter",
    "competitor", "risk", "opportunit", "trend", "market", "invest",
    "partnership", "strategy", "ceo", "jensen",
]


def _is_nvidia_relevant(question):
    q = question.lower()
    return any(k in q for k in _RELEVANCE_KEYWORDS)


# ----------------------------------------------------------------------
# deterministic briefing -> structured chat answer
# ----------------------------------------------------------------------
def _format_briefing_as_answer(b: dict) -> str:
    """
    Build the chat answer DETERMINISTICALLY from the structured briefing
    dict -- NOT from the model's free-text retelling, which varies run to
    run (sometimes a one-paragraph summary, sometimes a full dump). The
    briefing is schema-locked (fixed section counts: 3 opportunities, 3
    risks, 2 competitor, 2 trends, 3 recommendations, 3 action items), so
    formatting it here yields the SAME full structured breakdown every
    single turn. Also avoids the truncation bug (the model re-narrating
    the briefing used to run past REASON_MAX_NEW_TOKENS and cut off
    mid-sentence) and removes any chance of the retelling embellishing or
    misattributing -- every word now comes from the validated dict.
    """
    out = []

    if b.get("executive_summary"):
        out.append("**Executive Summary**")
        out.append(b["executive_summary"])

    def section(title, items, title_key, desc_key):
        if not items:
            return
        out.append(f"\n**{title}**")
        for i, item in enumerate(items, 1):
            sids = ", ".join(item.get("supporting_evidence", []))
            line = f"{i}. {item.get(title_key, '')}"
            if item.get(desc_key):
                line += f" — {item[desc_key]}"
            if sids:
                line += f" ({sids})"
            out.append(line)

    section("Key Opportunities", b.get("key_opportunities"), "opportunity", "business_impact")
    section("Key Risks", b.get("key_risks"), "risk", "why_it_matters")
    section("Competitor Activity", b.get("competitor_activity"), "competitor_activity", "strategic_meaning")
    section("Emerging Trends", b.get("emerging_trends"), "trend", "strategic_meaning")

    recs = b.get("strategic_recommendations", [])
    if recs:
        out.append("\n**Strategic Recommendations**")
        for i, rec in enumerate(recs, 1):
            sids = ", ".join(rec.get("supporting_evidence", []))
            line = (f"{i}. {rec.get('recommendation', '')} "
                    f"[Priority: {rec.get('priority', '—')}, Risk: {rec.get('risk_level', '—')}]")
            if rec.get("expected_impact"):
                line += f" — {rec['expected_impact']}"
            if sids:
                line += f" ({sids})"
            out.append(line)

    plan = b.get("ceo_action_plan", [])
    if plan:
        out.append("\n**CEO Action Plan**")
        for i, action in enumerate(plan, 1):
            out.append(f"{i}. {action}")

    return "\n".join(out)


# ----------------------------------------------------------------------
# graph state + nodes
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: list
    executed: list
    evidence: list
    final_answer: Optional[str]
    question: str
    fallback_used: bool
    plan: str
    validation: dict


_tokenizer = None
_model = None


def _ensure_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = get_base()
    return _tokenizer, _model


def reason_node(state: AgentState) -> dict:
    tokenizer, model = _ensure_model()

    prompt = tokenizer.apply_chat_template(
        state["messages"],
        tools=TOOL_SPECS,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=REASON_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)

    if RAW_DEBUG:
        print("\n----- RAW MODEL OUTPUT (before parsing) -----")
        print(raw[:1500])
        print("----- end raw output -----\n")

    calls = parse_tool_calls(raw)
    messages = state["messages"]

    if calls:
        tool_calls = []
        for c in calls:
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id": _gen_call_id(),
                "type": "function",
                "function": {"name": c["name"], "arguments": args},
            })
        messages.append({"role": "assistant", "tool_calls": tool_calls})
        return {"messages": messages}

    final = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": final})
    return {"messages": messages, "final_answer": final}


def act_node(state: AgentState) -> dict:
    messages = state["messages"]
    executed = state["executed"]
    evidence = state["evidence"]

    for tc in messages[-1]["tool_calls"]:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"] or {}

        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            result = f"Unknown tool: {name}"
        else:
            try:
                result = tool.invoke(args)
            except Exception as e:
                result = f"Tool {name} failed: {e}"

        executed.append({"name": name, "args": args})
        evidence.append({"tool": name, "result": result})
        messages.append({
            "role": "tool",
            "name": name,
            "content": result if isinstance(result, str) else str(result),
            "tool_call_id": tc["id"],
        })

    return {"messages": messages, "executed": executed, "evidence": evidence}


def fallback_act(state: AgentState) -> dict:
    """
    Safety net: model answered with ZERO tool calls on an NVIDIA-relevant
    question. Drop that ungrounded answer, force-call all four seekers,
    loop back to reason for a real, evidence-grounded answer.
    """
    messages = state["messages"]
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content"):
        messages = messages[:-1]

    executed = state["executed"]
    evidence = state["evidence"]
    topic = state["question"]

    tool_calls = [
        {
            "id": _gen_call_id(),
            "type": "function",
            "function": {"name": name, "arguments": {"topic": topic}},
        }
        for name in TOOLS_BY_NAME
    ]
    messages.append({"role": "assistant", "tool_calls": tool_calls})

    for tc in tool_calls:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        try:
            result = TOOLS_BY_NAME[name].invoke(args)
        except Exception as e:
            result = f"Tool {name} failed: {e}"

        executed.append({"name": name, "args": args})
        evidence.append({"tool": name, "result": result})
        messages.append({
            "role": "tool",
            "name": name,
            "content": result if isinstance(result, str) else str(result),
            "tool_call_id": tc["id"],
        })

    return {
        "messages": messages,
        "executed": executed,
        "evidence": evidence,
        "final_answer": None,
        "fallback_used": True,
    }


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return "act"
    if (not state["evidence"]
            and not state["fallback_used"]
            and _is_nvidia_relevant(state["question"])):
        return "fallback"
    return "end"


_app = None


def get_app():
    global _app
    if _app is None:
        g = StateGraph(AgentState)
        g.add_node("plan", plan_node)
        g.add_node("reason", reason_node)
        g.add_node("act", act_node)
        g.add_node("fallback", fallback_act)
        g.add_node("validate", validate_node)
        g.add_edge(START, "plan")
        g.add_edge("plan", "reason")
        g.add_conditional_edges("reason", route, {"act": "act", "fallback": "fallback", "end": "validate"})
        g.add_edge("act", "reason")
        g.add_edge("fallback", "reason")
        g.add_edge("validate", END)
        _app = g.compile()
    return _app


MAX_MEMORY_TURNS = 3


def ask(question: str, memory: list | None = None) -> dict:
    """
    Run one question through the agent. This is the REAL entry point --
    the dashboard calls this with the live user query. memory is an
    optional list of prior {"question","answer"} dicts owned by the
    caller; ask() never mutates it.
    """
    app = get_app()
    msgs = []
    if SYSTEM_PROMPT:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})

    for turn in (memory or [])[-MAX_MEMORY_TURNS:]:
        msgs.append({"role": "user", "content": turn["question"]})
        msgs.append({"role": "assistant", "content": turn["answer"]})

    msgs.append({"role": "user", "content": question})

    init: AgentState = {
        "messages": msgs,
        "executed": [],
        "evidence": [],
        "final_answer": None,
        "question": question,
        "fallback_used": False,
        "plan": "",
        "validation": {},
    }
    result = app.invoke(init, config={"recursion_limit": 12})

    briefing = None
    for ev in result.get("evidence", []):
        if ev.get("tool") == "generate_ceo_briefing":
            try:
                parsed = json.loads(ev["result"])
                if "error" not in parsed:
                    briefing = parsed
            except (json.JSONDecodeError, TypeError):
                pass

    final_answer = (result.get("final_answer") or "").strip()

    # When a structured briefing exists, build the chat answer FROM THE
    # DICT deterministically -- guaranteeing the full structured breakdown
    # (exec summary + opportunities + risks + competitor + trends +
    # recommendations + action plan) on EVERY turn, instead of the model's
    # inconsistent free-text retelling. Non-briefing turns (model chose a
    # single seeker) keep the model's grounded free-text answer.
    if briefing is not None:
        final_answer = _format_briefing_as_answer(briefing)

    if not final_answer:
        final_answer = "(no final answer produced)"

    return {
        "question": question,
        "answer": final_answer,
        "plan": result.get("plan", ""),
        "tool_calls": result.get("executed", []),
        "evidence": result.get("evidence", []),
        "briefing": briefing,
        "validation": result.get("validation", {}),
    }


def _print_result(result: dict) -> None:
    print(f"\nPLAN: {result['plan']}")
    print(f"\nTool calls made: {[c['name'] for c in result['tool_calls']]}")
    print(f"Briefing generated: {result['briefing'] is not None}")
    print(f"\nVALIDATION: {json.dumps(result['validation'])}")
    print(f"\nFINAL ANSWER:\n{result['answer']}\n")


if __name__ == "__main__":
    session_memory = []
    print("AI CEO agent ready. Type a question, or 'exit' / 'quit' to stop.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        print("=" * 70)
        print(f"(memory carried in: {len(session_memory)} prior turn(s))")
        print("=" * 70)
        result = ask(q, memory=session_memory)
        _print_result(result)
        session_memory.append({"question": q, "answer": result["answer"]})