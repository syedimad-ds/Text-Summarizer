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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0f1117;
    --surface:   #181c27;
    --surface2:  #1e2235;
    --border:    rgba(255,255,255,0.07);
    --accent:    #e8a44a;
    --accent2:   #c97b2e;
    --text:      #e8e9f0;
    --muted:     #7a7f94;
    --success:   #4ade80;
    --radius:    14px;
}

/* ── Global Reset ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, a.header-anchor,
.stDeployButton, .viewerBadge_container__1QSob { display: none !important; }

/* ── App container ── */
.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 1100px;
}

/* ── Page title area ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0 0 6px 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin-bottom: 2rem;
}
.accent-dot { color: var(--accent); }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.8rem 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-top: 1.4rem !important;
}

/* ── Sidebar logo/branding ── */
.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--text);
    margin-bottom: 4px;
}
.sidebar-tagline {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    color: var(--text) !important;
    font-size: 0.9rem !important;
}
[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem;
    color: var(--muted);
}

/* ── Sliders ── */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--accent) !important;
}
[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 0.82rem !important; }

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

/* ── Input card ── */
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,164,74,0.08) !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed rgba(232,164,74,0.3) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    text-align: center;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] * { color: var(--muted) !important; }

/* ── Primary button ── */
[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    color: #0f1117 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.5rem !important;
    transition: background 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 18px rgba(232,164,74,0.22) !important;
}
[data-testid="baseButton-primary"]:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px) !important;
}
[data-testid="baseButton-secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Stat cards ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 1.6rem 0 1.4rem;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    color: var(--text);
    line-height: 1;
}
.stat-value.accent { color: var(--accent); }

/* ── Summary output card ── */
.summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 22px 24px;
    font-size: 0.97rem;
    line-height: 1.75;
    color: var(--text);
    margin-bottom: 1.2rem;
}

/* ── Success / warning / error alerts ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p { color: var(--muted) !important; }

/* ── Metric (fallback) ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    padding: 14px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 8px; }

</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">✦ Summify</div>
    <div class="sidebar-tagline">AI-powered text summarizer</div>
    """, unsafe_allow_html=True)

    st.markdown("**Engine**")
    model_choice = st.radio(
        "engine",
        ["⚡ Fast Mode (flan-t5-small)", "🧠 High Accuracy (flan-t5-base)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Output Length**")
    summary_length = st.slider("Max tokens", 50, 500, 150, label_visibility="collapsed")
    st.caption(f"Max: {summary_length} tokens")
    min_length = st.slider("Min tokens", 10, 200, 30, label_visibility="collapsed")
    st.caption(f"Min: {min_length} tokens")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color: var(--muted); line-height:1.6;">
    Built with FLAN-T5 · Hugging Face<br>
    No data is stored or logged.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 3. HEADER
# ==========================================
st.markdown("""
<div class="hero-title">Intelligent Summarizer<span class="accent-dot">.</span></div>
<div class="hero-sub">Transform long documents into clear, concise summaries — instantly.</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. MODEL LOADER
# ==========================================
@st.cache_resource
def load_ai_engine(choice):
    model_name = "google/flan-t5-base" if "High Accuracy" in choice else "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner(f"Loading engine…"):
    tokenizer, model = load_ai_engine(model_choice)


# ==========================================
# 5. INPUT SECTION
# ==========================================
st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
input_type = st.radio("input_mode", ["📝 Paste Text", "📄 Upload PDF"],
                      horizontal=True, label_visibility="collapsed")

raw_text = ""

if input_type == "📝 Paste Text":
    raw_text = st.text_area(
        "text_input",
        placeholder="Paste your article, research paper, or document here…",
        height=220,
        label_visibility="collapsed"
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
        <div style="text-align:center; padding: 2rem; color: var(--muted); font-size:0.88rem;">
        📄 &nbsp; Drag and drop a PDF here, or click to browse
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 6. GENERATE BUTTON
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("✦ Generate Summary", type="primary", use_container_width=True)

st.markdown("---")


# ==========================================
# 7. SUMMARIZATION & OUTPUT
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
                summary_words = len(summary_text.split())
                compression = round((1 - summary_words / original_words) * 100, 1) if original_words else 0

                # ── Stat Cards ──
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="stat-label">Original</div>
                        <div class="stat-value">{original_words:,}</div>
                        <div style="font-size:0.75rem;color:var(--muted);margin-top:2px">words</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Summary</div>
                        <div class="stat-value accent">{summary_words:,}</div>
                        <div style="font-size:0.75rem;color:var(--muted);margin-top:2px">words</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Compressed</div>
                        <div class="stat-value">{compression}%</div>
                        <div style="font-size:0.75rem;color:var(--muted);margin-top:2px">reduction</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Time</div>
                        <div class="stat-value">{time_taken}s</div>
                        <div style="font-size:0.75rem;color:var(--muted);margin-top:2px">inference</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Summary Output ──
                st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-card">{summary_text}</div>', unsafe_allow_html=True)

                # ── Download ──
                st.download_button(
                    label="↓ Download as .txt",
                    data=f"ORIGINAL: {original_words} words\nSUMMARY: {summary_words} words\nCOMPRESSION: {compression}%\nTIME: {time_taken}s\n\n---\n\n{summary_text}",
                    file_name="summary.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
