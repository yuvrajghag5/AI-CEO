"""
Executive Intelligence Dashboard  (Box 6)  —  live agent edition

Live chat calling agent.react_agent.ask() drives the five intelligence
panels; corpus files drive Company Overview + Sentiment. No evidence.json
(that belonged to the retired rule-based engine).

LAYOUT
  Left  : live chat. The user's question and the working spinner render
          INSIDE the chat box, where the answer lands. Session memory
          persists across turns. Each answer carries a plan/tools/
          validation expander -- the ReAct trace.
  Right : the 7 rubric panels as tabs, updating after each turn.

DATA SOURCES
  Corpus-level, read once at load (cached):
    - data/cleaned/clean_documents.json    -> Company Overview, Source mix
    - data/cleaned/sentiment_analysis.json -> Sentiment Analysis
  Live, per chat turn (result['briefing']):
    - key_opportunities          -> Opportunity Monitor
    - key_risks                  -> Risk Monitor
    - competitor_activity, emerging_trends -> Market Intelligence
    - strategic_recommendations  -> Strategic Recommendations
    - executive_summary, ceo_action_plan, _sources -> CEO Briefing

PERSISTENCE
  Each turn overwrites data/evidence/ceo_report.json (latest turn only).

Place at: dashboard/dashboard.py
Launch  : python main.py   (Streamlit on :8501 + ngrok tunnel)
"""
import sys
import os

# project root on sys.path so `config`, `agent`, etc. import regardless
# of the directory Streamlit launches from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import Counter
from datetime import datetime, timezone, timedelta

import streamlit as st
import plotly.express as px
import pandas as pd

CLEAN_DOCS_FILE = "data/cleaned/clean_documents.json"
SENTIMENT_FILE = "data/cleaned/sentiment_analysis.json"
CEO_REPORT_FILE = "data/evidence/ceo_report.json"
PIPELINE_META_FILE = "data/pipeline_meta.json"
COMPANY_NAME = "NVIDIA"
INDUSTRY = "Semiconductors / AI Computing"

# sentiment trend: only plot the recent window. The corpus contains some
# old community posts (HackerNews threads going back years) whose dates
# would otherwise stretch the x-axis to 2012 and make the "live
# intelligence" trend look broken. 90 days keeps it current and honest.
SENTIMENT_TREND_DAYS = 90

# ---------------------------------------------------------------- #
# Page config + theme
# ---------------------------------------------------------------- #
st.set_page_config(
    page_title="AI-CEO Strategic Intelligence",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NVIDIA_GREEN = "#76B900"
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#2a313c"
TEXT_MUTED = "#9aa4b2"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background-color: {DARK_BG}; }}

section[data-testid="stSidebar"] {{
    background-color: {PANEL_BG};
    border-right: 1px solid {BORDER};
}}

h1, h2, h3 {{ color: #f0f3f7 !important; font-weight: 800 !important; }}
.accent {{ color: {NVIDIA_GREEN}; }}

.metric-card {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}}
.metric-label {{
    color: {TEXT_MUTED}; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 4px;
}}
.metric-value {{ color: #f0f3f7; font-size: 1.8rem; font-weight: 800; }}

.evidence-card {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-left: 4px solid {NVIDIA_GREEN};
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
    color: #d6dce3;
}}
.evidence-card.risk {{ border-left-color: #e05252; }}
.evidence-card.trend {{ border-left-color: #4aa8e0; }}
.evidence-card.competitor {{ border-left-color: #d8a13a; }}
.evidence-card h4 {{ margin: 0 0 6px 0; color: #f0f3f7; }}

.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.badge-high {{ background-color: rgba(224,82,82,0.18); color: #e05252; }}
.badge-medium {{ background-color: rgba(216,161,58,0.18); color: #d8a13a; }}
.badge-low {{ background-color: rgba(118,185,0,0.18); color: {NVIDIA_GREEN}; }}

.source-pill {{
    display: inline-block; background-color: rgba(118,185,0,0.12);
    color: {NVIDIA_GREEN}; border: 1px solid rgba(118,185,0,0.3);
    border-radius: 6px; padding: 1px 8px; margin: 2px 4px 2px 0;
    font-size: 0.75rem; font-family: monospace;
}}

.briefing-block {{
    background-color: {PANEL_BG}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 24px 26px; line-height: 1.6; color: #d6dce3;
}}
.empty-state {{
    color: {TEXT_MUTED}; text-align: center; padding: 44px 16px;
    border: 1px dashed {BORDER}; border-radius: 10px;
}}
a {{ color: {NVIDIA_GREEN} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_last_run():
    """Read the collection pipeline's last-run timestamp for the KPI.

    Written by collectors/run_pipeline.py on successful completion. Falls
    back to a dash if the pipeline hasn't run since this was added.
    """
    meta = load_json(PIPELINE_META_FILE)
    if not meta or not meta.get("last_run"):
        return "—"
    raw = meta["last_run"]
    try:
        return (
            datetime.fromisoformat(raw.replace("Z", "+00:00"))
            .strftime("%b %d, %Y · %H:%M UTC")
        )
    except (ValueError, AttributeError):
        return raw[:16].replace("T", " ")


def badge_class(level):
    return {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(level, "badge-medium")


def render_metric(label, value):
    st.markdown(
        f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_sids(ids):
    """Live briefing cites S1/S2... citation IDs (not doc_ids)."""
    if not ids:
        return ""
    return "".join(f'<span class="source-pill">{sid}</span>' for sid in ids)


def empty(msg):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- #
# Live agent  (loaded once; holds the model via the model.py singleton)
# ---------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_agent():
    from agent.react_agent import ask
    return ask


def save_report(result):
    """Overwrite ceo_report.json with the latest turn's full result."""
    try:
        os.makedirs(os.path.dirname(CEO_REPORT_FILE), exist_ok=True)
        payload = dict(result)
        payload["_saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(CEO_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as e:
        st.warning(f"Could not write ceo_report.json: {e}")


# ---------------------------------------------------------------- #
# Corpus data (read once)
# ---------------------------------------------------------------- #
documents = load_json(CLEAN_DOCS_FILE) or []
sentiment_entries = load_json(SENTIMENT_FILE) or []


# ================================================================ #
# PANELS
# ================================================================ #
def panel_overview():
    st.title("Company Overview")
    sources_count = len({d.get("source") for d in documents})
    last_run = load_pipeline_last_run()
    cols = st.columns(5)
    with cols[0]:
        render_metric("Company", COMPANY_NAME)
    with cols[1]:
        render_metric("Industry", INDUSTRY)
    with cols[2]:
        render_metric("Documents", f"{len(documents):,}")
    with cols[3]:
        render_metric("Data Sources", sources_count)
    with cols[4]:
        render_metric("Last Updated", last_run)

    st.markdown("#### Source Breakdown")
    source_counts = Counter(d.get("source") for d in documents)
    if source_counts:
        df = pd.DataFrame(source_counts.items(), columns=["Source", "Documents"])
        fig = px.bar(df, x="Source", y="Documents", color="Source",
                     color_discrete_sequence=[NVIDIA_GREEN, "#4aa8e0", "#d8a13a", "#e05252"])
        fig.update_layout(plot_bgcolor=PANEL_BG, paper_bgcolor=DARK_BG,
                          font_color="#d6dce3", showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty("No documents found — run the collection pipeline first.")


def panel_market_intel(result):
    st.title("Market Intelligence")
    b = (result or {}).get("briefing")
    if not b:
        empty("Ask a strategic question to populate market intelligence.")
        return

    st.markdown("#### Competitor Activity")
    for item in b.get("competitor_activity", []):
        st.markdown(
            f"""<div class="evidence-card competitor">
                    <h4>{item.get('competitor_activity','')}</h4>
                    <p>{item.get('strategic_meaning','')}</p>
                    {render_sids(item.get('supporting_evidence'))}
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### Emerging Technologies & Trends")
    for item in b.get("emerging_trends", []):
        st.markdown(
            f"""<div class="evidence-card trend">
                    <h4>{item.get('trend','')}</h4>
                    <p>{item.get('strategic_meaning','')}</p>
                    {render_sids(item.get('supporting_evidence'))}
                </div>""",
            unsafe_allow_html=True,
        )


def panel_opportunities(result):
    st.title("Opportunity Monitor")
    b = (result or {}).get("briefing")
    if not b:
        empty("No opportunities yet — ask the agent a question.")
        return
    for item in b.get("key_opportunities", []):
        st.markdown(
            f"""<div class="evidence-card">
                    <h4>{item.get('opportunity','')}</h4>
                    <p>{item.get('business_impact','')}</p>
                    <span style="color:{TEXT_MUTED};font-size:0.8rem;">Evidence: </span>
                    {render_sids(item.get('supporting_evidence'))}
                </div>""",
            unsafe_allow_html=True,
        )


def panel_risks(result):
    st.title("Risk Monitor")
    b = (result or {}).get("briefing")
    if not b:
        empty("No risks yet — ask the agent a question.")
        return
    for item in b.get("key_risks", []):
        st.markdown(
            f"""<div class="evidence-card risk">
                    <h4>{item.get('risk','')}</h4>
                    <p>{item.get('why_it_matters','')}</p>
                    <span style="color:{TEXT_MUTED};font-size:0.8rem;">Evidence: </span>
                    {render_sids(item.get('supporting_evidence'))}
                </div>""",
            unsafe_allow_html=True,
        )


def panel_sentiment():
    st.title("Sentiment Analysis")
    if not sentiment_entries:
        empty("No sentiment data found — run sentiment.py first.")
        return

    label_counts = Counter(e.get("sentiment_label") for e in sentiment_entries)
    cols = st.columns(3)
    for col, label in zip(cols, ["positive", "neutral", "negative"]):
        with col:
            render_metric(label.capitalize(), label_counts.get(label, 0))

    color_map = {"positive": NVIDIA_GREEN, "neutral": "#9aa4b2", "negative": "#e05252"}
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Overall Distribution")
        df = pd.DataFrame(label_counts.items(), columns=["Sentiment", "Count"])
        fig = px.pie(df, names="Sentiment", values="Count",
                     color="Sentiment", color_discrete_map=color_map, hole=0.5)
        fig.update_layout(paper_bgcolor=DARK_BG, font_color="#d6dce3", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Sentiment by Source")
        df_src = pd.DataFrame(sentiment_entries)
        if "source" in df_src.columns:
            pivot = df_src.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
            fig2 = px.bar(pivot, x="source", y="count", color="sentiment_label",
                          color_discrete_map=color_map, barmode="stack")
            fig2.update_layout(plot_bgcolor=PANEL_BG, paper_bgcolor=DARK_BG,
                               font_color="#d6dce3", height=300)
            st.plotly_chart(fig2, use_container_width=True)

    # --- trend over time, RECENT WINDOW ONLY ---
    # old community posts in the corpus carry dates going back years; we
    # clip to the last SENTIMENT_TREND_DAYS so the "live" trend reflects
    # the current monitoring window rather than a 14-year sparse line.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SENTIMENT_TREND_DAYS)).strftime("%Y-%m-%d")
    by_day = {}
    for e in sentiment_entries:
        ds = (e.get("published_date") or "")[:10]
        if ds and ds >= cutoff:
            by_day.setdefault(ds, []).append(e.get("sentiment_score", 0.0))

    st.markdown(f"#### Sentiment Trend (daily average · last {SENTIMENT_TREND_DAYS} days)")
    if by_day:
        trend = pd.DataFrame(
            [(day, sum(v) / len(v)) for day, v in sorted(by_day.items())],
            columns=["Date", "Avg sentiment"],
        )
        fig3 = px.line(trend, x="Date", y="Avg sentiment", markers=True,
                       color_discrete_sequence=[NVIDIA_GREEN])
        fig3.update_layout(plot_bgcolor=PANEL_BG, paper_bgcolor=DARK_BG,
                           font_color="#d6dce3", height=300)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        empty(f"No sentiment records in the last {SENTIMENT_TREND_DAYS} days.")


def panel_recommendations(result):
    st.title("Strategic Recommendations")
    b = (result or {}).get("briefing")
    if not b:
        empty("No recommendations yet — ask the agent a question.")
        return
    for rec in b.get("strategic_recommendations", []):
        st.markdown(
            f"""<div class="evidence-card">
                    <h4>{rec.get('recommendation','')}</h4>
                    <span class="badge {badge_class(rec.get('priority'))}">Priority: {rec.get('priority')}</span>
                    &nbsp;&nbsp;
                    <span class="badge {badge_class(rec.get('risk_level'))}">Risk: {rec.get('risk_level')}</span>
                    <br><br>
                    <b>Expected Impact:</b> {rec.get('expected_impact','')}<br><br>
                    <span style="color:{TEXT_MUTED};font-size:0.8rem;">Evidence: </span>
                    {render_sids(rec.get('supporting_evidence'))}
                </div>""",
            unsafe_allow_html=True,
        )


def panel_ceo_briefing(result):
    st.title("CEO Briefing")
    b = (result or {}).get("briefing")
    if not b:
        empty("The executive summary appears here once you ask a question.")
        return

    if result.get("question"):
        st.caption(result["question"])

    st.markdown("### Executive Summary")
    st.markdown(f'<div class="briefing-block">{b.get("executive_summary","")}</div>',
                unsafe_allow_html=True)

    st.markdown("### CEO Action Plan")
    for i, action in enumerate(b.get("ceo_action_plan", []), 1):
        st.markdown(f'<div class="evidence-card"><b class="accent">{i}.</b> {action}</div>',
                    unsafe_allow_html=True)

    # validation report (from validate_node)
    val = result.get("validation", {})
    if val:
        ok = val.get("passed")
        color = NVIDIA_GREEN if ok else "#e05252"
        st.markdown(
            f'<p style="color:{TEXT_MUTED};">Validation: '
            f'<span style="color:{color};font-weight:700;">'
            f'{val.get("citations_verified",0)}/{val.get("citations_checked",0)} '
            f'citations verified — {"passed" if ok else "flagged"}</span></p>',
            unsafe_allow_html=True,
        )
        for flag in val.get("flagged", []):
            st.markdown(f'<p style="color:#e05252;font-size:0.85rem;">⚠ {flag}</p>',
                        unsafe_allow_html=True)

    sources = b.get("_sources", {})
    with st.expander(f"📚 Sources Used ({len(sources)})"):
        for sid, src in sources.items():
            st.markdown(
                f"**[{sid}]** {src.get('title','')} "
                f"<span style='color:{TEXT_MUTED};'>({src.get('source','')})</span>",
                unsafe_allow_html=True,
            )
            if src.get("url"):
                st.caption(src["url"])
            st.divider()


# ================================================================ #
# MAIN  — chat left, 7 panels as tabs right
# ================================================================ #
def main():
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest" not in st.session_state:
        st.session_state.latest = {}

    st.markdown(
        f"<h1>AI <span class='accent'>CEO</span> — Strategic Intelligence "
        f"<span style='color:{TEXT_MUTED};font-weight:400;'>· {COMPANY_NAME}</span></h1>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.35], gap="large")

    # ---------------- LEFT: live chat ----------------
    with left:
        st.markdown("#### Ask the CEO agent")
        st.markdown(
            f'<p style="color:{TEXT_MUTED};">e.g. "If you were the CEO today, '
            f'what would you do next and why?"</p>',
            unsafe_allow_html=True,
        )

        chat_box = st.container(height=540)
        with chat_box:
            for turn in st.session_state.history:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])
                    if turn["role"] == "assistant" and turn.get("plan"):
                        with st.expander("plan · tools · validation"):
                            st.markdown(f"**Plan:** {turn['plan']}")
                            st.markdown(f"**Tools called:** {', '.join(turn.get('tools', [])) or '—'}")
                            st.markdown(f"**Validation:** `{json.dumps(turn.get('validation', {}))}`")

        prompt = st.chat_input("Ask a strategic question…")
        if prompt:
            # Render the user's message AND the working spinner INSIDE the
            # chat box -- in the assistant bubble position, where the
            # answer will land after rerun. The spinner no longer sits
            # below the input bar.
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Planning → retrieving → reasoning → validating…"):
                        ask = get_agent()
                        result = ask(prompt, memory=st.session_state.memory)

            save_report(result)
            st.session_state.latest = result
            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.memory.append({"question": prompt, "answer": result["answer"]})
            st.session_state.history.append({
                "role": "assistant",
                "content": result["answer"],
                "plan": result.get("plan", ""),
                "tools": [c["name"] for c in result.get("tool_calls", [])],
                "validation": result.get("validation", {}),
            })
            st.rerun()

    # ---------------- RIGHT: 7 panels ----------------
    with right:
        result = st.session_state.latest
        tabs = st.tabs([
            "Overview", "Market Intel", "Opportunities",
            "Risks", "Sentiment", "Recommendations", "CEO Briefing",
        ])
        with tabs[0]:
            panel_overview()
        with tabs[1]:
            panel_market_intel(result)
        with tabs[2]:
            panel_opportunities(result)
        with tabs[3]:
            panel_risks(result)
        with tabs[4]:
            panel_sentiment()
        with tabs[5]:
            panel_recommendations(result)
        with tabs[6]:
            panel_ceo_briefing(result)


if __name__ == "__main__":
    main()