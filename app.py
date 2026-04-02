import streamlit as st
import time
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. PAGE SETUP & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Summify", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;1,500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:            #0c0e18;
    --surface-solid: #131622;
    --surface2:      #1a1e30;
    --border:        rgba(255,255,255,0.07);
    --border-lit:    rgba(232,164,74,0.25);
    --accent:        #e8a44a;
    --accent2:       #c97b2e;
    --text:          #e8e9f0;
    --muted:         #6b7194;
    --radius:        14px;
}

/* ════════════════════════════════════════
   KEYFRAMES
════════════════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes orbDrift1 {
    0%   { transform: translate(0,0) scale(1); }
    33%  { transform: translate(60px,-40px) scale(1.08); }
    66%  { transform: translate(-30px,50px) scale(0.95); }
    100% { transform: translate(0,0) scale(1); }
}
@keyframes orbDrift2 {
    0%   { transform: translate(0,0) scale(1); }
    40%  { transform: translate(-70px,30px) scale(1.05); }
    80%  { transform: translate(40px,-60px) scale(0.97); }
    100% { transform: translate(0,0) scale(1); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes borderPulse {
    0%,100% { border-left-color: var(--accent); box-shadow: -3px 0 14px rgba(232,164,74,0.2); }
    50%     { border-left-color: #f0bc72;       box-shadow: -3px 0 28px rgba(232,164,74,0.45); }
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(18px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
}
@keyframes sidebarSlide {
    from { opacity: 0; transform: translateX(-14px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes accentBlink {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.4; }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes btnPulse {
    0%,100% { box-shadow: 0 4px 20px rgba(232,164,74,0.25); }
    50%     { box-shadow: 0 4px 32px rgba(232,164,74,0.5); }
}

/* ── Accessibility ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* ════════════════════════════════════════
   GLOBAL
════════════════════════════════════════ */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, a.header-anchor,
.stDeployButton, .viewerBadge_container__1QSob { display: none !important; }

/* App container */
.block-container {
    padding: 2.5rem 3rem 5rem !important;
    max-width: 1100px;
    position: relative;
}

/* ════════════════════════════════════════
   AMBIENT ORBS
════════════════════════════════════════ */
.stApp::before, .stApp::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(90px);
}
.stApp::before {
    width: 520px; height: 520px;
    background: radial-gradient(circle, rgba(232,164,74,0.09) 0%, transparent 70%);
    top: -120px; left: -100px;
    animation: orbDrift1 22s ease-in-out infinite;
}
.stApp::after {
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(100,120,255,0.06) 0%, transparent 70%);
    bottom: 80px; right: -80px;
    animation: orbDrift2 28s ease-in-out infinite;
}

/* ════════════════════════════════════════
   HERO
════════════════════════════════════════ */
.hero-wrap {
    animation: fadeUp 0.7s cubic-bezier(.22,.68,0,1.2) both;
    position: relative; z-index: 1;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(232,164,74,0.08);
    border: 1px solid rgba(232,164,74,0.2);
    border-radius: 100px;
    padding: 4px 12px;
    margin-bottom: 14px;
}
.hero-badge-dot {
    width: 5px; height: 5px;
    background: var(--accent);
    border-radius: 50%;
    animation: accentBlink 2s ease-in-out infinite;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -1px;
    line-height: 1.05;
    margin: 0 0 10px 0;
}
.hero-title em {
    font-style: italic;
    background: linear-gradient(90deg, var(--accent), #f5c97a, var(--accent));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.97rem;
    font-weight: 300;
    margin-bottom: 2.2rem;
    animation: fadeUp 0.7s 0.15s cubic-bezier(.22,.68,0,1.2) both;
}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--surface-solid) !important;
    border-right: 1px solid var(--border) !important;
    animation: sidebarSlide 0.6s cubic-bezier(.22,.68,0,1.2) both;
}
[data-testid="stSidebar"] > div { padding-top: 1.8rem !important; }
.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    color: var(--text);
    margin-bottom: 3px;
    letter-spacing: -0.3px;
}
.sidebar-tagline { font-size: 0.76rem; color: var(--muted); margin-bottom: 1.6rem; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-top: 1.4rem !important;
}
[data-testid="stRadio"] label {
    color: var(--text) !important;
    font-size: 0.88rem !important;
    transition: color 0.2s;
}
[data-testid="stRadio"] label:hover { color: var(--accent) !important; }
[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 0.8rem !important; }

/* ════════════════════════════════════════
   SECTION LABEL
════════════════════════════════════════ */
.section-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
    animation: fadeIn 0.5s 0.3s both;
}

/* ════════════════════════════════════════
   DIVIDER
════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.8rem 0 !important;
}

/* ════════════════════════════════════════
   TEXTAREA
════════════════════════════════════════ */
[data-testid="stTextArea"] textarea {
    background: var(--surface-solid) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    padding: 16px 18px !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    animation: fadeUp 0.6s 0.25s cubic-bezier(.22,.68,0,1.2) both;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,164,74,0.1), 0 0 24px rgba(232,164,74,0.06) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* ════════════════════════════════════════
   FILE UPLOADER
════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--surface-solid) !important;
    border: 1px dashed var(--border-lit) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: border-color 0.25s, background 0.25s !important;
    animation: fadeUp 0.6s 0.25s both;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: rgba(232,164,74,0.03) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] * { color: var(--muted) !important; }

/* ════════════════════════════════════════
   GENERATE BUTTON
   Multi-layer selectors to beat Streamlit's
   own stylesheet on both desktop and mobile
════════════════════════════════════════ */
.stButton > button,
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"],
button[data-testid="baseButton-primary"] {
    position: relative !important;
    overflow: hidden !important;
    background: var(--accent) !important;
    background-color: var(--accent) !important;
    color: #0c0e18 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(232,164,74,0.25) !important;
    animation: fadeUp 0.6s 0.4s cubic-bezier(.22,.68,0,1.2) both,
               btnPulse 2.5s 1.2s ease-in-out 3 !important;
}
/* Shimmer sweep */
.stButton > button::after,
[data-testid="baseButton-primary"]::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.22) 50%, transparent 70%);
    background-size: 200% 100%;
    background-position: -200% center;
    pointer-events: none;
}
.stButton > button:hover::after,
[data-testid="baseButton-primary"]:hover::after {
    animation: shimmer 0.55s ease forwards;
}
.stButton > button:hover,
[data-testid="baseButton-primary"]:hover {
    background: #edaf58 !important;
    background-color: #edaf58 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(232,164,74,0.4) !important;
}
.stButton > button:active,
[data-testid="baseButton-primary"]:active {
    transform: translateY(0) scale(0.97) !important;
    box-shadow: 0 2px 10px rgba(232,164,74,0.2) !important;
}

/* Download button — secondary, must come AFTER primary rules */
[data-testid="baseButton-secondary"],
button[data-testid="baseButton-secondary"] {
    background: var(--surface2) !important;
    background-color: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    box-shadow: none !important;
    animation: none !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--border-lit) !important;
    background: rgba(232,164,74,0.06) !important;
    background-color: rgba(232,164,74,0.06) !important;
    transform: none !important;
}

/* ════════════════════════════════════════
   STAT CARDS
════════════════════════════════════════ */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 1.6rem 0 1.4rem;
}
.stat-card {
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    opacity: 0;
    animation: cardIn 0.5s cubic-bezier(.22,.68,0,1.2) forwards;
    transition: border-color 0.25s, transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    border-color: var(--border-lit);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.stat-card:nth-child(1) { animation-delay: 0.05s; }
.stat-card:nth-child(2) { animation-delay: 0.13s; }
.stat-card:nth-child(3) { animation-delay: 0.21s; }
.stat-card:nth-child(4) { animation-delay: 0.29s; }
.stat-label {
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--muted);
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.85rem; color: var(--text); line-height: 1;
}
.stat-value.accent { color: var(--accent); }
.stat-unit { font-size: 0.73rem; color: var(--muted); margin-top: 1px; }

/* ════════════════════════════════════════
   SUMMARY CARD
════════════════════════════════════════ */
.summary-card {
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 24px 26px;
    font-size: 0.97rem;
    line-height: 1.8;
    color: var(--text);
    margin-bottom: 1.4rem;
    animation: fadeUp 0.6s 0.1s cubic-bezier(.22,.68,0,1.2) both,
               borderPulse 3s 0.8s ease-in-out 3;
}

/* ════════════════════════════════════════
   MOBILE SETTINGS PANEL
   Rendered by Python but hidden on desktop.
   CSS flips display:block on mobile.
════════════════════════════════════════ */
.mobile-settings-wrap {
    display: none;
}
.mobile-settings-inner {
    background: var(--surface-solid);
    border: 1px solid var(--border-lit);
    border-radius: var(--radius);
    padding: 14px 16px 8px;
    margin-bottom: 1.4rem;
    animation: fadeUp 0.5s 0.1s both;
}
.mobile-settings-title {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}

/* ════════════════════════════════════════
   ALERTS / SPINNER
════════════════════════════════════════ */
[data-testid="stAlert"] {
    background: var(--surface-solid) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    animation: fadeIn 0.35s both;
}
[data-testid="stSpinner"] p   { color: var(--muted) !important; font-size: 0.88rem !important; }
[data-testid="stSpinner"] svg { stroke: var(--accent) !important; }

/* ════════════════════════════════════════
   SCROLLBAR
════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ════════════════════════════════════════
   RADIO — horizontal input-mode picker
════════════════════════════════════════ */
[data-testid="stRadio"][data-horizontal="true"] label {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    transition: border-color 0.2s, background 0.2s !important;
    font-size: 0.88rem !important;
}
[data-testid="stRadio"][data-horizontal="true"] label:hover {
    border-color: var(--border-lit) !important;
    background: rgba(232,164,74,0.04) !important;
}

/* ════════════════════════════════════════
   MOBILE  ≤ 768 px
════════════════════════════════════════ */
@media (max-width: 768px) {

    /* ── 1. Nuke sidebar and every possible toggle element ── */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    div[data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarNav"],
    [data-testid="collapsedControl"],
    .stSidebarCollapsedControl,
    button[kind="header"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"],
    button[aria-label="open sidebar"],
    button[aria-label="close sidebar"] {
        display:        none !important;
        visibility:     hidden !important;
        width:          0 !important;
        min-width:      0 !important;
        max-width:      0 !important;
        overflow:       hidden !important;
        pointer-events: none !important;
    }

    /* ── 2. Remove the reserved sidebar margin ── */
    .stApp > .stMain,
    .stApp [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {
        margin-left:  0 !important;
        padding-left: 0 !important;
    }

    /* ── 3. Tighter padding ── */
    .block-container {
        padding: 1.4rem 1rem 3rem !important;
    }

    /* ── 4. Responsive hero text ── */
    .hero-title  { font-size: 2.1rem !important; letter-spacing: -0.5px !important; }
    .hero-sub    { font-size: 0.88rem !important; }

    /* ── 5. Show the in-page settings panel ── */
    .mobile-settings-wrap { display: block !important; }

    /* ── 6. Stats 2×2 grid ── */
    .stat-row { grid-template-columns: repeat(2, 1fr) !important; gap: 10px !important; }
    .stat-value { font-size: 1.45rem !important; }

    /* ── 7. Amber button — highest possible specificity.
              Streamlit mobile injects background:rgb(255,75,75)
              via an emotion-cache class. We beat it with
              html body .stApp chaining + !important. ── */
    html body .stApp .stButton > button,
    html body .stApp .stButton > button[kind="primary"],
    html body .stApp [data-testid="baseButton-primary"],
    html body .stApp button[data-testid="baseButton-primary"] {
        background:       var(--accent) !important;
        background-color: var(--accent) !important;
        color:            #0c0e18 !important;
        border:           none !important;
        box-shadow:       0 4px 20px rgba(232,164,74,0.3) !important;
    }
    html body .stApp .stButton > button:hover,
    html body .stApp [data-testid="baseButton-primary"]:hover {
        background:       #edaf58 !important;
        background-color: #edaf58 !important;
    }

    /* Keep download button dark on mobile */
    html body .stApp [data-testid="baseButton-secondary"],
    html body .stApp button[data-testid="baseButton-secondary"] {
        background:       var(--surface2) !important;
        background-color: var(--surface2) !important;
        color:            var(--text) !important;
        box-shadow:       none !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SESSION STATE
#    model_choice persists across reruns so
#    both sidebar (desktop) and inline panel
#    (mobile) stay in sync.
# ==========================================
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "⚡ Fast Mode (flan-t5-small)"
if "summary_length" not in st.session_state:
    st.session_state.summary_length = 150
if "min_length" not in st.session_state:
    st.session_state.min_length = 30


# ==========================================
# 3. SIDEBAR  (desktop)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">✦ Summify</div>
    <div class="sidebar-tagline">AI-powered text summarizer</div>
    """, unsafe_allow_html=True)

    st.markdown("**Engine**")
    desktop_model = st.radio(
        "desktop_engine",
        ["⚡ Fast Mode (flan-t5-small)", "🧠 High Accuracy (flan-t5-base)"],
        index=0 if "small" in st.session_state.model_choice else 1,
        label_visibility="collapsed",
    )
    st.session_state.model_choice = desktop_model

    st.markdown("---")
    st.markdown("**Output Length**")
    desktop_max = st.slider("Desktop max", 50, 500, st.session_state.summary_length,
                             label_visibility="collapsed")
    st.caption(f"Max: {desktop_max} tokens")
    st.session_state.summary_length = desktop_max

    desktop_min = st.slider("Desktop min", 10, 200, st.session_state.min_length,
                             label_visibility="collapsed")
    st.caption(f"Min: {desktop_min} tokens")
    st.session_state.min_length = desktop_min

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:var(--muted); line-height:1.6;">
    Built with FLAN-T5 · Hugging Face<br>
    No data is stored or logged.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 4. HEADER
# ==========================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge"><span class="hero-badge-dot"></span> FLAN-T5 · Abstractive AI</div>
    <div class="hero-title">Intelligent <em>Summarizer</em></div>
</div>
<div class="hero-sub">Transform long documents into clear, concise summaries — instantly.</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. MOBILE SETTINGS PANEL
#    Hidden on desktop via CSS.
#    Gives mobile users full control over
#    model and length without needing sidebar.
# ==========================================
st.markdown("""
<div class="mobile-settings-wrap">
    <div class="mobile-settings-inner">
        <div class="mobile-settings-title">⚙️ Settings</div>
    </div>
</div>
""", unsafe_allow_html=True)

# These widgets are always rendered by Python (needed for state),
# but CSS hides the entire .mobile-settings-wrap on desktop,
# so desktop users only ever see the sidebar controls.
mobile_model = st.radio(
    "mobile_engine",
    ["⚡ Fast Mode (flan-t5-small)", "🧠 High Accuracy (flan-t5-base)"],
    index=0 if "small" in st.session_state.model_choice else 1,
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.model_choice = mobile_model

mobile_max = st.slider(
    "Mobile max length",
    50, 500, st.session_state.summary_length,
    label_visibility="collapsed",
)
st.session_state.summary_length = mobile_max

mobile_min = st.slider(
    "Mobile min length",
    10, 200, st.session_state.min_length,
    label_visibility="collapsed",
)
st.session_state.min_length = mobile_min

# Resolve final values from session state
model_choice   = st.session_state.model_choice
summary_length = st.session_state.summary_length
min_length     = st.session_state.min_length


# ==========================================
# 6. MODEL LOADER
# ==========================================
@st.cache_resource
def load_ai_engine(choice):
    model_name = "google/flan-t5-base" if "High Accuracy" in choice else "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading engine…"):
    tokenizer, model = load_ai_engine(model_choice)


# ==========================================
# 7. INPUT SECTION
# ==========================================
st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
input_type = st.radio(
    "input_mode",
    ["📝 Paste Text", "📄 Upload PDF"],
    horizontal=True,
    label_visibility="collapsed",
)

raw_text = ""

if input_type == "📝 Paste Text":
    raw_text = st.text_area(
        "text_input",
        placeholder="Paste your article, research paper, or document here…",
        height=220,
        label_visibility="collapsed",
    )
else:
    uploaded_file = st.file_uploader("pdf_upload", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + " "
            st.success(f"✅ PDF loaded — {len(raw_text.split()):,} words extracted")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:var(--muted); font-size:0.88rem;">
        📄 &nbsp; Drag and drop a PDF here, or click to browse
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 8. GENERATE BUTTON
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("✦ Generate Summary", type="primary", use_container_width=True)

st.markdown("---")


# ==========================================
# 9. SUMMARIZATION & OUTPUT
# ==========================================
if generate:
    word_count = len(raw_text.split())
    if word_count < 40:
        st.warning("⚠️ Please provide at least 40 words to summarize.")
    else:
        with st.spinner("Analyzing and summarizing…"):
            start_time = time.time()
            try:
                prompt = (
                    "Strictly summarize the following text using only the provided information. "
                    "Do not add outside knowledge:\n\n" + raw_text
                )
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=summary_length,
                    min_length=min_length,
                    length_penalty=1.5,
                    num_beams=4,
                    early_stopping=False,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.15,
                )
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                time_taken = round(time.time() - start_time, 2)

                original_words = len(raw_text.split())
                summary_words  = len(summary_text.split())
                compression    = round((1 - summary_words / original_words) * 100, 1) if original_words else 0

                # ── Stat Cards ──
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="stat-label">Original</div>
                        <div class="stat-value">{original_words:,}</div>
                        <div class="stat-unit">words</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Summary</div>
                        <div class="stat-value accent">{summary_words:,}</div>
                        <div class="stat-unit">words</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Compressed</div>
                        <div class="stat-value">{compression}%</div>
                        <div class="stat-unit">reduction</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Time</div>
                        <div class="stat-value">{time_taken}s</div>
                        <div class="stat-unit">inference</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Summary Output ──
                st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-card">{summary_text}</div>', unsafe_allow_html=True)

                # ── Download ──
                st.download_button(
                    label="↓ Download as .txt",
                    data=(
                        f"ORIGINAL: {original_words} words\n"
                        f"SUMMARY:  {summary_words} words\n"
                        f"COMPRESSION: {compression}%\n"
                        f"TIME: {time_taken}s\n\n---\n\n{summary_text}"
                    ),
                    file_name="summary.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
