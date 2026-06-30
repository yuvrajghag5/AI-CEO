# """
# The four seeker tools  (Box 3)

# Each tool is a thin wrapper around engine.seek(). They take a `topic`
# (the thing the user is asking about) and return that category's evidence
# as readable text.

# In Option 3, the agent (Box 4) calls ALL FOUR of these for each user
# question, then synthesises them into a CEO briefing (Box 5). So these
# four are the agent's "gather evidence" hands.

# You can test them yourself right now by running this file — it asks you
# for a topic and fires all four seekers on it, exactly the way the agent
# will. That's the user-input -> evidence connection, by hand.

# Place at: agent/tools.py
# Run standalone: python -m agent.tools
# """
# from langchain_core.tools import tool

# import json
# from engine.engine import seek
# from agent.briefing import generate_ceo_briefing_dict


# def _format(result):
#     """Turn a seek() result dict into readable text for the agent."""
#     if not result["evidence"]:
#         return (f"No relevant {result['category']} evidence found "
#                 f"for '{result['topic']}'.")

#     header = (f"{result['category'].upper()} for '{result['topic']}' "
#               f"(confidence: {result['confidence']}, impact: {result['impact']})")
#     lines = [header]
#     for ev in result["evidence"]:
#         lines.append(f"  [{ev['source']}] {ev['title']} — {ev['excerpt'][:200]}")
#     return "\n".join(lines)


# @tool
# def risk_seeker(topic: str) -> str:
#     """Find strategic risks facing NVIDIA related to the given topic. Use when the user asks about threats, regulatory issues, competition, supply chain problems, or downside scenarios."""
#     return _format(seek("risks", topic))


# @tool
# def opportunity_seeker(topic: str) -> str:
#     """Find strategic opportunities for NVIDIA related to the given topic. Use when the user asks about growth, new markets, partnerships, product launches, or upside scenarios."""
#     return _format(seek("opportunities", topic))


# @tool
# def trend_seeker(topic: str) -> str:
#     """Find emerging trends relevant to NVIDIA related to the given topic. Use when the user asks about industry shifts, customer behaviour changes, or technology adoption trends."""
#     return _format(seek("trends", topic))


# @tool
# def competitor_activity_seeker(topic: str) -> str:
#     """Find competitor activity relevant to NVIDIA related to the given topic. Use when the user asks what AMD, Intel, Qualcomm or other competitors are doing, market share shifts, or competitive positioning."""
#     return _format(seek("competitor_activity", topic))


# @tool
# def generate_ceo_briefing(topic: str) -> str:
#     """Generate a complete structured CEO briefing for NVIDIA on the given topic, including executive summary, key opportunities, key risks, competitor activity, emerging trends, strategic recommendations, and a concrete action plan, all grounded in real evidence. Use this for a broad strategic question or when the user explicitly wants a full briefing or recommendation, not for narrow single-fact lookups."""
#     briefing = generate_ceo_briefing_dict(topic)
#     return json.dumps(briefing, ensure_ascii=False)


# # the agent (Box 4) can call any subset of these per question
# ALL_TOOLS = [
#     risk_seeker,
#     opportunity_seeker,
#     trend_seeker,
#     competitor_activity_seeker,
#     generate_ceo_briefing,
# ]


# if __name__ == "__main__":
#     # Test by hand: type a topic, watch all four seekers fire on it.
#     # This is exactly what the agent will do automatically later.
#     topic = input("Enter a topic (e.g. 'data center', 'supply chain', 'AI chips'): ").strip()
#     if not topic:
#         topic = "data center"
#         print(f"(empty — using '{topic}')")

#     print()
#     # .invoke({"topic": topic}) is how you call a LangChain @tool directly
#     for seeker in ALL_TOOLS:
#         print("=" * 60)
#         print(seeker.invoke({"topic": topic}))
#         print()



"""
The four seeker tools  (Box 3)

Each tool is a thin wrapper around engine.seek(). They take a `topic`
(the thing the user is asking about) and return that category's evidence
as readable text.

In Option 3, the agent (Box 4) calls ALL FOUR of these for each user
question, then synthesises them into a CEO briefing (Box 5). So these
four are the agent's "gather evidence" hands.

You can test them yourself right now by running this file — it asks you
for a topic and fires all four seekers on it, exactly the way the agent
will. That's the user-input -> evidence connection, by hand.

Place at: agent/tools.py
Run standalone: python -m agent.tools
"""
from langchain_core.tools import tool

import json
from engine.engine import seek
from agent.briefing import generate_ceo_briefing_dict


def _format(result):
    """Turn a seek() result dict into readable text for the agent.

    Includes the REAL url for each piece of evidence. Without it, the
    model has no genuine link to cite and will sometimes invent a
    plausible-looking but fake one (observed: same fabricated newsapi.org
    URL repeated across multiple distinct claims). Always give it the
    real thing instead.
    """
    if not result["evidence"]:
        return (f"No relevant {result['category']} evidence found "
                f"for '{result['topic']}'.")

    header = (f"{result['category'].upper()} for '{result['topic']}' "
              f"(confidence: {result['confidence']}, impact: {result['impact']})")
    lines = [header]
    for ev in result["evidence"]:
        url = ev.get("url") or "(no url available)"
        lines.append(f"  [{ev['source']}] {ev['title']} — {ev['excerpt'][:200]} (url: {url})")
    return "\n".join(lines)


@tool
def risk_seeker(topic: str) -> str:
    """Find strategic risks facing NVIDIA related to the given topic. Use when the user asks about threats, regulatory issues, competition, supply chain problems, or downside scenarios."""
    return _format(seek("risks", topic))


@tool
def opportunity_seeker(topic: str) -> str:
    """Find strategic opportunities for NVIDIA related to the given topic. Use when the user asks about growth, new markets, partnerships, product launches, or upside scenarios."""
    return _format(seek("opportunities", topic))


@tool
def trend_seeker(topic: str) -> str:
    """Find emerging trends relevant to NVIDIA related to the given topic. Use when the user asks about industry shifts, customer behaviour changes, or technology adoption trends."""
    return _format(seek("trends", topic))


@tool
def competitor_activity_seeker(topic: str) -> str:
    """Find competitor activity relevant to NVIDIA related to the given topic. Use when the user asks what AMD, Intel, Qualcomm or other competitors are doing, market share shifts, or competitive positioning."""
    return _format(seek("competitor_activity", topic))


@tool
def generate_ceo_briefing(topic: str) -> str:
    """Generate a complete structured CEO briefing for NVIDIA on the given topic, including executive summary, key opportunities, key risks, competitor activity, emerging trends, strategic recommendations, and a concrete action plan, all grounded in real evidence. Use this for a broad strategic question or when the user explicitly wants a full briefing or recommendation, not for narrow single-fact lookups."""
    briefing = generate_ceo_briefing_dict(topic)
    return json.dumps(briefing, ensure_ascii=False)


# the agent (Box 4) can call any subset of these per question
ALL_TOOLS = [
    risk_seeker,
    opportunity_seeker,
    trend_seeker,
    competitor_activity_seeker,
    generate_ceo_briefing,
]


if __name__ == "__main__":
    # Test by hand: type a topic, watch all four seekers fire on it.
    # This is exactly what the agent will do automatically later.
    topic = input("Enter a topic (e.g. 'data center', 'supply chain', 'AI chips'): ").strip()
    if not topic:
        topic = "data center"
        print(f"(empty — using '{topic}')")

    print()
    # .invoke({"topic": topic}) is how you call a LangChain @tool directly
    for seeker in ALL_TOOLS:
        print("=" * 60)
        print(seeker.invoke({"topic": topic}))
        print()