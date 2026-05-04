"""
BIS Standards Recommendation Engine — Streamlit Frontend
Run: streamlit run streamlit_app.py
"""

import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

CATEGORY_COLORS = {
    "Cement":        "#0D9488",
    "Steel":         "#3B82F6",
    "Concrete":      "#8B5CF6",
    "Aggregates":    "#F59E0B",
    "Masonry":       "#EF4444",
    "Tiles":         "#EC4899",
    "Timber":        "#84CC16",
    "Glass":         "#06B6D4",
    "Paints":        "#F97316",
    "Waterproofing": "#14B8A6",
    "Pipes":         "#6366F1",
    "Doors":         "#A855F7",
    "General":       "#64748B",
}

EXAMPLE_QUERIES = [
    "OPC 53 grade cement for high strength concrete",
    "TMT Fe500 bars for concrete reinforcement",
    "Concrete mix design for M30 grade",
    "Coarse and fine aggregate specification",
    "Sulphate resisting cement for underground foundations",
    "Portland pozzolana cement fly ash based",
    "Ready mix concrete code of practice",
    "GGBS ground granulated blast furnace slag",
    "Silica fume for high performance concrete",
    "Prestressed concrete post tensioning",
]

st.set_page_config(
    page_title="BIS Standards Finder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0A1628 !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #060F1E !important;
    border-right: 1px solid #1E3A5F !important;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

/* Header */
.bis-header {
    background: linear-gradient(135deg, #0D2137 0%, #0F3D38 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.bis-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #0D9488 !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.5px;
}
.bis-header p {
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* Search box */
.stTextArea textarea {
    background-color: #0F1E30 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.2) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0D9488, #0F766E) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0F766E, #115E59) !important;
    transform: translateY(-1px) !important;
}

/* Result card */
.result-card {
    background: #0F1E30;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    transition: border-color 0.2s ease;
}
.result-card:hover {
    border-color: #0D9488;
}
.result-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.result-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0D9488;
    margin-bottom: 0.3rem;
}
.result-title {
    font-size: 0.95rem;
    font-weight: 500;
    color: #CBD5E1;
    margin-bottom: 0.6rem;
    line-height: 1.4;
}
.result-rationale {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.5;
    border-top: 1px solid #1E3A5F;
    padding-top: 0.6rem;
    margin-top: 0.4rem;
}
.category-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.score-bar-bg {
    background: #1E3A5F;
    border-radius: 4px;
    height: 4px;
    margin-top: 0.8rem;
}
.score-bar-fill {
    background: linear-gradient(90deg, #0D9488, #06B6D4);
    border-radius: 4px;
    height: 4px;
}

/* Expansion chip */
.expansion-chip {
    display: inline-block;
    background: #0F3D38;
    border: 1px solid #0D9488;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #0D9488;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* Metric card */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    background: #0F1E30;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0D9488;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}

/* Sidebar example button */
.stButton.example-btn > button {
    background: transparent !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 6px !important;
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.4rem 0.8rem !important;
    margin-bottom: 0.2rem !important;
}
.stButton.example-btn > button:hover {
    border-color: #0D9488 !important;
    color: #0D9488 !important;
    transform: none !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ BIS Finder")
    st.markdown("<p style='color:#64748B;font-size:0.82rem;margin-top:-0.5rem'>BIS × SS Hackathon 2025</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Try an example query:**")
    for eq in EXAMPLE_QUERIES:
        if st.button(eq, key=f"ex_{eq}", use_container_width=True):
            st.session_state.query = eq
            st.rerun()

    st.markdown("---")

    # Knowledge base stats
    try:
        resp = requests.get(f"{API_URL}/standards", timeout=2)
        if resp.ok:
            data = resp.json()
            st.markdown(f"**Knowledge Base**")
            st.markdown(f"<p style='color:#0D9488;font-family:JetBrains Mono;font-size:1.2rem;font-weight:700'>{data['total']} Standards</p>", unsafe_allow_html=True)
            cats = data.get("categories", [])
            st.markdown(f"<p style='color:#64748B;font-size:0.8rem'>{len(cats)} categories · BIS SP 21</p>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<p style='color:#EF4444;font-size:0.82rem'>⚠ Backend offline</p>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("**Recent searches**")
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"<p style='color:#64748B;font-size:0.8rem'>• {h[:45]}…</p>", unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='bis-header'>
  <h1>🏗 BIS Standards Recommendation Engine</h1>
  <p>Describe your product or material — get the right Bureau of Indian Standards in seconds.</p>
</div>
""", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])

with col_input:
    query = st.text_area(
        label="Product Description",
        value=st.session_state.query,
        placeholder='e.g. "OPC 53 grade cement for high strength concrete" or "TMT Fe500 bars for RCC"',
        height=80,
        label_visibility="collapsed",
        key="query_input",
    )

with col_btn:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search", use_container_width=True)

# ── Search logic ──────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    st.session_state.query = query
    if query not in st.session_state.history:
        st.session_state.history.append(query)

    with st.spinner("Running hybrid retrieval…"):
        try:
            resp = requests.post(
                f"{API_URL}/recommend",
                json={"query": query, "top_k": 5},
                timeout=10,
            )
            if resp.ok:
                st.session_state.results = resp.json()
            else:
                st.error(f"API error: {resp.status_code}")
                st.session_state.results = None
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to backend. Make sure `uvicorn app:app --port 8000` is running.")
            st.session_state.results = None

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    data = st.session_state.results
    results = data["results"]
    latency = data["latency_seconds"]
    expanded_terms = data.get("expanded_terms", [])

    # Metrics row
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-val'>{len(results)}</div>
            <div class='metric-label'>Standards Found</div>
        </div>
        <div class='metric-card'>
            <div class='metric-val'>{latency:.3f}s</div>
            <div class='metric-label'>Latency</div>
        </div>
        <div class='metric-card'>
            <div class='metric-val'>{results[0]['standard_id'] if results else '—'}</div>
            <div class='metric-label'>Top Result</div>
        </div>
        <div class='metric-card'>
            <div class='metric-val'>{results[0]['category'] if results else '—'}</div>
            <div class='metric-label'>Category</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Query expansion chips
    if expanded_terms:
        chips = "".join(f"<span class='expansion-chip'>{t}</span>" for t in expanded_terms)
        st.markdown(f"<div style='margin-bottom:1rem'><span style='color:#64748B;font-size:0.82rem;margin-right:0.5rem'>Query expanded:</span>{chips}</div>", unsafe_allow_html=True)

    # Result cards
    medals = ["🥇", "🥈", "🥉", "4th", "5th"]
    for i, r in enumerate(results):
        cat_color = CATEGORY_COLORS.get(r["category"], "#64748B")
        score_pct = int(r["score"] * 100)

        st.markdown(f"""
        <div class='result-card'>
            <div class='result-rank'>{medals[i] if i < 3 else f'#{i+1}'} &nbsp;·&nbsp; Rank {i+1}</div>
            <span class='category-badge' style='background:{cat_color}22;color:{cat_color};border:1px solid {cat_color}44'>{r['category']}</span>
            <div class='result-id'>{r['standard_id']}</div>
            <div class='result-title'>{r['title']}</div>
            <div class='result-rationale'>📋 {r['rationale']}</div>
            <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{min(score_pct*3, 100)}%'></div>
            </div>
            <p style='color:#475569;font-size:0.72rem;margin:0.3rem 0 0 0;font-family:JetBrains Mono'>relevance score: {r['score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#1E3A5F'>
        <div style='font-size:3rem;margin-bottom:1rem'>🏗️</div>
        <p style='color:#334155;font-size:1rem'>Enter a product description above or pick an example from the sidebar</p>
        <p style='color:#1E3A5F;font-size:0.82rem;margin-top:0.5rem'>Covers Cement · Steel · Concrete · Aggregates · Masonry · and more</p>
    </div>
    """, unsafe_allow_html=True)
