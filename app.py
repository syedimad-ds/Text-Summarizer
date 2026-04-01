import streamlit as st
import time
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. PAGE SETUP & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Text Summarizer", page_icon="🚀", layout="wide")

# Yeh CSS code saare headings (title, sidebar, etc.) se link/copy icon ko hide kar dega
st.markdown(
    """
    <style>
    /* Hide the header anchor links (copy icon) */
    a.header-anchor {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 Text Summarizer")
st.markdown("Summarize any text or PDF document instantly using advanced AI.")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ AI Engine Settings")
model_choice = st.sidebar.radio(
    "Choose AI Engine:",
    ["⚡ Fast Mode (flan-t5-small)", "🧠 High Accuracy (flan-t5-base)"]
)

st.sidebar.markdown("---")
st.sidebar.header("📏 Length Controller")
summary_length = st.sidebar.slider("Max Summary Length", 50, 500, 150)
min_length = st.sidebar.slider("Min Summary Length", 10, 200, 30)

# ==========================================
# 3. AI ENGINE LOADER
# ==========================================
@st.cache_resource
def load_ai_engine(choice):
    model_name = "google/flan-t5-base" if "High Accuracy" in choice else "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner(f"Loading {model_choice} engine... Please wait."):
    tokenizer, model = load_ai_engine(model_choice)

# ==========================================
# 4. INPUT HANDLERS
# ==========================================
st.markdown("### 📥 Choose your input method:")
input_type = st.radio("", ["📝 Paste Text", "📄 Upload PDF"], horizontal=True)
raw_text = ""

if input_type == "📝 Paste Text":
    raw_text = st.text_area("Paste your long text here...", height=200)

elif input_type == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text: raw_text += text + " "
            st.success("✅ PDF Extracted Successfully!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

# ==========================================
# 5. SUMMARIZER PROCESS
# ==========================================
st.markdown("---")
if st.button("✨ Generate Summary", type="primary", use_container_width=True):
    if len(raw_text.split()) < 40:
        st.warning("⚠️ Please provide at least 40 words to summarize.")
    else:
        with st.spinner("AI is analyzing and summarizing your text..."):
            start_time = time.time()
            try:
                # Prompt to prevent AI hallucination
                prompt = "Strictly summarize the following text using only the provided information. Do not add outside knowledge: \n\n" + raw_text
                inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                
                summary_ids = model.generate(
                    inputs.input_ids, 
                    max_length=summary_length, 
                    min_length=min_length, 
                    length_penalty=1.5, 
                    num_beams=4, 
                    early_stopping=True,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.15
                )
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                time_taken = round(time.time() - start_time, 2)
                
                original_words = len(raw_text.split())
                summary_words = len(summary_text.split())
                compression_ratio = round((1 - (summary_words / original_words)) * 100, 1) if original_words > 0 else 0

                # Analytics Heading Removed completely as requested
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📝 Original Words", original_words)
                col2.metric("🎯 Summary Words", summary_words)
                col3.metric("📉 Compression", f"{compression_ratio}%")
                col4.metric("⏱️ Time", f"{time_taken} sec")

                st.markdown("### 💡 AI Summary")
                st.info(summary_text)

                st.download_button(
                    label="📥 Download TXT",
                    data=f"ORIGINAL: {original_words} words\nSUMMARY: {summary_words} words\n\n{summary_text}",
                    file_name="Text_Summary.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
