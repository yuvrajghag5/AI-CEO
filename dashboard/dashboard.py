# RUN :-  streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false



import sys
import os
 
# Add project root to sys.path so `config.paths` etc. are importable
# regardless of which directory this script is launched from (Streamlit
# sets the script's own folder as the import root, not the project root).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import json
from collections import Counter
from datetime import datetime
 
import streamlit as st
import plotly.express as px
import pandas as pd
 
CLEAN_DOCS_FILE = "data/cleaned/clean_documents.json"
SENTIMENT_FILE = "data/cleaned/sentiment_analysis.json"
EVIDENCE_FILE = "data/evidence/evidence.json"
CEO_REPORT_FILE = "data/evidence/ceo_report.json"
COMPANY_NAME = "NVIDIA"
INDUSTRY = "Semiconductors / AI Computing"
 
# ---------------------------------------------------------------- #
# Page config + theme
# ---------------------------------------------------------------- #
st.set_page_config(
    page_title="AI-CEO Strategic Intelligence",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
NVIDIA_GREEN = "#76B900"
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#2a313c"
TEXT_MUTED = "#9aa4b2"
 
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
 
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
 
.stApp {{
    background-color: {DARK_BG};
}}
 
section[data-testid="stSidebar"] {{
    background-color: {PANEL_BG};
    border-right: 1px solid {BORDER};
}}
 
h1, h2, h3 {{
    color: #f0f3f7 !important;
    font-weight: 800 !important;
}}
 
.accent {{
    color: {NVIDIA_GREEN};
}}
 
.metric-card {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}}
 
.metric-label {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}}
 
.metric-value {{
    color: #f0f3f7;
    font-size: 1.8rem;
    font-weight: 800;
}}
 
.evidence-card {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-left: 4px solid {NVIDIA_GREEN};
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
}}
 
.evidence-card.risk {{
    border-left-color: #e05252;
}}
 
.evidence-card.trend {{
    border-left-color: #4aa8e0;
}}
 
.evidence-card.competitor {{
    border-left-color: #d8a13a;
}}
 
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
 
.badge-high {{ background-color: rgba(224,82,82,0.18); color: #e05252; }}
.badge-medium {{ background-color: rgba(216,161,58,0.18); color: #d8a13a; }}
.badge-low {{ background-color: rgba(118,185,0,0.18); color: {NVIDIA_GREEN}; }}
 
.source-pill {{
    display: inline-block;
    background-color: rgba(118,185,0,0.12);
    color: {NVIDIA_GREEN};
    border: 1px solid rgba(118,185,0,0.3);
    border-radius: 6px;
    padding: 1px 8px;
    margin: 2px 4px 2px 0;
    font-size: 0.75rem;
    font-family: monospace;
}}
 
.briefing-block {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 24px 26px;
    line-height: 1.6;
    color: #d6dce3;
}}
 
a {{ color: {NVIDIA_GREEN} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------- #
# Data loading
# ---------------------------------------------------------------- #
@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
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
 
 
def render_source_pills(ids):
    if not ids:
        return ""
    return "".join(f'<span class="source-pill">{sid}</span>' for sid in ids)
 
 
# ---------------------------------------------------------------- #
# Load all data once
# ---------------------------------------------------------------- #
documents = load_json(CLEAN_DOCS_FILE) or []
sentiment_entries = load_json(SENTIMENT_FILE) or []
evidence = load_json(EVIDENCE_FILE) or {}
ceo_report = load_json(CEO_REPORT_FILE) or {}
 
# ---------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"## 🟢 <span class='accent'>{COMPANY_NAME}</span>", unsafe_allow_html=True)
    st.caption("AI Strategic Intelligence Agent")
    st.divider()
    section = st.radio(
        "Navigate",
        [
            "Company Overview",
            "Market Intelligence",
            "Opportunity Monitor",
            "Risk Monitor",
            "Sentiment Analysis",
            "Strategic Recommendations",
            "CEO Briefing",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Last updated: {ceo_report.get('generated_at', 'N/A')[:19].replace('T', ' ')}")
 
 
# ================================================================ #
# SECTION 1: Company Overview
# ================================================================ #
if section == "Company Overview":
    st.title("Company Overview")
 
    sources_count = len({doc.get("source") for doc in documents})
    cols = st.columns(4)
    with cols[0]:
        render_metric("Company", COMPANY_NAME)
    with cols[1]:
        render_metric("Industry", INDUSTRY)
    with cols[2]:
        render_metric("Documents Collected", len(documents))
    with cols[3]:
        render_metric("Data Sources", sources_count)
 
    st.markdown("#### Source Breakdown")
    source_counts = Counter(doc.get("source") for doc in documents)
    if source_counts:
        df = pd.DataFrame(source_counts.items(), columns=["Source", "Documents"])
        fig = px.bar(df, x="Source", y="Documents", color="Source",
                     color_discrete_sequence=[NVIDIA_GREEN, "#4aa8e0", "#d8a13a", "#e05252"])
        fig.update_layout(plot_bgcolor=PANEL_BG, paper_bgcolor=DARK_BG, font_color="#d6dce3", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents found yet — run the collection pipeline first.")
 
 
# ================================================================ #
# SECTION 2: Market Intelligence
# ================================================================ #
elif section == "Market Intelligence":
    st.title("Market Intelligence")
 
    tab1, tab2, tab3 = st.tabs(["📰 Recent News", "⚔️ Competitor Activity", "🚀 Emerging Technologies"])
 
    with tab1:
        recent_docs = sorted(documents, key=lambda d: d.get("published_date") or "", reverse=True)[:10]
        for doc in recent_docs:
            st.markdown(
                f"""<div class="evidence-card">
                        <b>{doc.get('title', 'Untitled')}</b><br>
                        <span style="color:{TEXT_MUTED}; font-size:0.85rem;">
                            {doc.get('source', '').upper()} · {doc.get('published_date', 'N/A')}
                        </span>
                    </div>""",
                unsafe_allow_html=True,
            )
 
    with tab2:
        for item in evidence.get("competitor_activity", []):
            st.markdown(
                f"""<div class="evidence-card competitor">
                        <b>{item.get('title')}</b>
                        <span class="badge {badge_class(item.get('impact'))}">{item.get('impact')}</span><br><br>
                        {render_source_pills([e.get('doc_id') for e in item.get('evidence', [])])}
                    </div>""",
                unsafe_allow_html=True,
            )
 
    with tab3:
        for item in evidence.get("trends", []):
            st.markdown(
                f"""<div class="evidence-card trend">
                        <b>{item.get('title')}</b>
                        <span class="badge {badge_class(item.get('impact'))}">{item.get('impact')}</span><br><br>
                        {render_source_pills([e.get('doc_id') for e in item.get('evidence', [])])}
                    </div>""",
                unsafe_allow_html=True,
            )
 
 
# ================================================================ #
# SECTION 3: Opportunity Monitor
# ================================================================ #
elif section == "Opportunity Monitor":
    st.title("Opportunity Monitor")
    opportunities = evidence.get("opportunities", [])
 
    if not opportunities:
        st.info("No opportunities found — run engine.py first.")
    for item in opportunities:
        with st.container():
            st.markdown(
                f"""<div class="evidence-card">
                        <h4 style="margin:0;">{item.get('title')}</h4>
                        <span class="badge {badge_class(item.get('impact'))}">Impact: {item.get('impact')}</span>
                        &nbsp;&nbsp;<span style="color:{TEXT_MUTED};">Confidence: {item.get('confidence')}</span>
                        <br><br>
                        {render_source_pills([e.get('doc_id') for e in item.get('evidence', [])])}
                    </div>""",
                unsafe_allow_html=True,
            )
            with st.expander("View supporting evidence"):
                for ev in item.get("evidence", []):
                    st.markdown(f"**[{ev.get('source')}]** {ev.get('title')}")
                    st.caption(ev.get("excerpt", "")[:300])
                    if ev.get("url"):
                        st.markdown(f"[Source link]({ev.get('url')})")
                    st.divider()
 
 
# ================================================================ #
# SECTION 4: Risk Monitor
# ================================================================ #
elif section == "Risk Monitor":
    st.title("Risk Monitor")
    risks = evidence.get("risks", [])
 
    if not risks:
        st.info("No risks found — run engine.py first.")
    for item in risks:
        with st.container():
            st.markdown(
                f"""<div class="evidence-card risk">
                        <h4 style="margin:0;">{item.get('title')}</h4>
                        <span class="badge {badge_class(item.get('impact'))}">Severity: {item.get('impact')}</span>
                        &nbsp;&nbsp;<span style="color:{TEXT_MUTED};">Confidence: {item.get('confidence')}</span>
                        <br><br>
                        {render_source_pills([e.get('doc_id') for e in item.get('evidence', [])])}
                    </div>""",
                unsafe_allow_html=True,
            )
            with st.expander("View supporting evidence"):
                for ev in item.get("evidence", []):
                    st.markdown(f"**[{ev.get('source')}]** {ev.get('title')}")
                    st.caption(ev.get("excerpt", "")[:300])
                    if ev.get("url"):
                        st.markdown(f"[Source link]({ev.get('url')})")
                    st.divider()
 
 
# ================================================================ #
# SECTION 5: Sentiment Analysis
# ================================================================ #
elif section == "Sentiment Analysis":
    st.title("Sentiment Analysis")
 
    if not sentiment_entries:
        st.info("No sentiment data found — run sentiment.py first.")
    else:
        label_counts = Counter(e.get("sentiment_label") for e in sentiment_entries)
        cols = st.columns(3)
        for col, label in zip(cols, ["positive", "neutral", "negative"]):
            with col:
                render_metric(label.capitalize(), label_counts.get(label, 0))
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.markdown("#### Overall Sentiment Distribution")
            df = pd.DataFrame(label_counts.items(), columns=["Sentiment", "Count"])
            color_map = {"positive": NVIDIA_GREEN, "neutral": "#9aa4b2", "negative": "#e05252"}
            fig = px.pie(df, names="Sentiment", values="Count",
                         color="Sentiment", color_discrete_map=color_map, hole=0.5)
            fig.update_layout(paper_bgcolor=DARK_BG, font_color="#d6dce3")
            st.plotly_chart(fig, use_container_width=True)
 
        with col2:
            st.markdown("#### Sentiment by Source")
            df_source = pd.DataFrame(sentiment_entries)
            if "source" in df_source.columns:
                pivot = df_source.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
                fig2 = px.bar(pivot, x="source", y="count", color="sentiment_label",
                              color_discrete_map=color_map, barmode="stack")
                fig2.update_layout(plot_bgcolor=PANEL_BG, paper_bgcolor=DARK_BG, font_color="#d6dce3")
                st.plotly_chart(fig2, use_container_width=True)
 
 
# ================================================================ #
# SECTION 6: Strategic Recommendations
# ================================================================ #
elif section == "Strategic Recommendations":
    st.title("Strategic Recommendations")
    recommendations = ceo_report.get("strategic_recommendations", [])
 
    if not recommendations:
        st.info("No recommendations found — run agent.py first.")
    for rec in recommendations:
        st.markdown(
            f"""<div class="evidence-card">
                    <h4 style="margin:0;">{rec.get('recommendation')}</h4>
                    <span class="badge {badge_class(rec.get('priority'))}">Priority: {rec.get('priority')}</span>
                    &nbsp;&nbsp;
                    <span class="badge {badge_class(rec.get('risk_level'))}">Risk: {rec.get('risk_level')}</span>
                    <br><br>
                    <b>Expected Impact:</b> {rec.get('expected_impact')}<br><br>
                    {render_source_pills(rec.get('supporting_evidence', []))}
                </div>""",
            unsafe_allow_html=True,
        )
 
 
# ================================================================ #
# SECTION 7: CEO Briefing
# ================================================================ #
elif section == "CEO Briefing":
    st.title("CEO Briefing")
    st.caption(ceo_report.get("user_question", ""))
 
    if not ceo_report:
        st.info("No CEO report found — run agent.py first.")
    else:
        st.markdown("### Executive Summary")
        st.markdown(f"""<div class="briefing-block">{ceo_report.get('executive_summary', '')}</div>""",
                    unsafe_allow_html=True)
 
        st.markdown("### CEO Action Plan")
        for i, action in enumerate(ceo_report.get("ceo_action_plan", []), 1):
            st.markdown(
                f"""<div class="evidence-card"><b>{i}.</b> {action}</div>""",
                unsafe_allow_html=True,
            )
 
        with st.expander(f"📚 Sources Used ({len(ceo_report.get('sources_used', []))})"):
            for src in ceo_report.get("sources_used", []):
                score = src.get("score")
                score_text = f" · relevance {score:.2f}" if score is not None else ""
                st.markdown(
                    f"**[{src.get('source_id')}]** {src.get('title')} "
                    f"<span style='color:{TEXT_MUTED};'>({src.get('source')}{score_text})</span>",
                    unsafe_allow_html=True,
                )
                if src.get("url"):
                    st.caption(src.get("url"))
                st.divider()
 