![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

# 🚀 Universal AI Text Summarizer 
**Powered by FLAN-T5 (Large Language Models) & Streamlit**

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://text-summarizer-k2.streamlit.app/)

A high-performance, abstractive summarization engine designed to transform dense academic papers, legal documents, and long-form articles into concise, human-like summaries. Unlike traditional tools that simply "cut and paste" sentences, this app utilizes **Sequence-to-Sequence (Seq2Seq)** architectures to rephrase and restructure information.

---

## 🛠️ Tech Stack
- **AI Model:** Google FLAN-T5 (Small & Base variants)
- **Framework:** Streamlit (UI & State Management)
- **Libraries:** Hugging Face Transformers, PyTorch, PyPDF2 (Vectorized Extraction)
- **Deployment:** GitHub & Streamlit Community Cloud

---

## 📂 Repository Structure

```text
Text-Summarizer/
│
├── app.py                      # Core Logic (UI, Inference Pipeline, CSS Injection)
├── requirements.txt            # Dependency management for Cloud Deployment
├── Text_Summarizer.ipynb       # Research & Development / Local Testing Notebook
└── README.md                   # Technical documentation
```
---

# ✨ Key Features & Engineering Optimizations

## 1. Dual-Engine Inference Pipeline
- **Fast Mode (flan-t5-small):** Optimized for low-latency summarization of news articles and short blogs.  
- **High Accuracy (flan-t5-base):** Leverages deeper context windows for complex document structures.  

## 2. Intelligent Document Vectorization
- Utilizes PyPDF2 for robust text extraction.  
- Handles multi-page PDFs with vectorized cleanup to ensure minimal noise during tokenization.  

## 3. Anti-Hallucination & Stuttering Control
- **N-Gram Blocking:** Implements `no_repeat_ngram_size=4` to eliminate repetitive loops common in Seq2Seq models.  
- **Repetition Penalty:** Tuned at `1.15` to strike a balance between factual naming (like "DNA" or "NASA") and creative rephrasing.  
- **Length Penalty:** Set at `1.5` to ensure summaries are informative without being verbose.  

## 4. Real-Time Analytical Dashboard
- **Compression Metrics:** Instant calculation of % reduction in text volume.  
- **Inference Latency Tracking:** Monitors model performance (seconds/request) in real-time.  
- **Word Density Analysis:** Compares original vs. summarized token counts.  

---

# 🧠 Design Philosophy: Why FLAN-T5?

We chose an **Abstractive Approach** over Extractive Summarization because modern NLP demands context.  

- **Extractive (Old School):** Picks existing sentences. Often misses the "big picture."  
- **Abstractive (Our Choice):** Acts like a human brain. It reads, understands, and rewrites.  

**FLAN-T5:** We selected Instruction-Tuned T5 because it excels at zero-shot summarization without requiring massive GPU VRAM, making it the perfect candidate for Edge/Cloud deployment.  

---

# 📉 Technical Constraints & Shortcomings

## 1. Context Window Limitations
- **Problem:** The model has a max token limit (approx. 512-1024 tokens).  
- **Impact:** Extremely long books or 50+ page PDFs require "chunking" (planned for v2.0).  

## 2. "Prior Knowledge" Bias (Hallucinations)
- **Problem:** LLMs sometimes add outside facts (e.g., assuming a "Central Bank" is the "Federal Reserve").  
- **Solution:** We implemented strict prompt-engineering instructions to keep the model grounded in the provided text.  

## 3. Computational Overhead
- **Problem:** flan-t5-base can be slow on free-tier CPUs.  
- **Optimization:** Cached model loading using `@st.cache_resource` to ensure the model stays in memory after the first run.  

---

# 🚀 Local Setup & Installation

## Clone the Repo
- git clone https://github.com/syedimad-ds/Text-Summarizer.git
- pip install -r requirements.txt
- streamlit run app.py

---

# 📊 Demonstrates Strong Proficiency In:

- Large Language Model (LLM) integration via Hugging Face Hub.  
- Parameter Tuning (Beam search, Temperature, Repetition Penalty).  
- Clean UI/UX Design with Custom CSS Injection.  
- End-to-End AI Deployment (Local dev to Public Cloud).  

---

# 👨‍💻 Author

**Syed Imad Muzaffar**  
🎓 3rd Year B.E. Student — Artificial Intelligence & Data Science  
