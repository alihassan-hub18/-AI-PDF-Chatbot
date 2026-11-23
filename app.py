import io
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

st.set_page_config(page_title="Beginner RAG Agent", page_icon="📘", layout="wide")

st.markdown("""
<style>
.stApp{background:#0b1220;color:#e5e7eb}
[data-testid="stSidebar"]{background:#111827}
.hero{padding:24px;border:1px solid #26364f;border-radius:18px;
background:linear-gradient(120deg,#172554,#111827);margin-bottom:18px}
.hero h1{color:#fff;margin:0}.hero p{color:#cbd5e1;margin-bottom:0}
.panel{padding:14px;border:1px solid #29384f;border-radius:14px;background:#111827}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📘 Smart Document RAG Assistant</h1><p>Beginner-friendly Retrieval-Augmented Generation · TF-IDF · Cosine Similarity · Gemini</p></div>', unsafe_allow_html=True)

if "docs" not in st.session_state:
    st.session_state.docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "matrix" not in st.session_state:
    st.session_state.matrix = None

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API key", type="password")
    model_name = st.text_input("Gemini model", value="gemini-3.8-flash")
    chunk_words = st.slider("Words per chunk", 100, 500, 220, 20)
    top_k = st.slider("Relevant chunks (Top-K)", 1, 8, 4)
    st.caption("Your API key is used for requests and is not saved in this app.")
    if st.button("Reset knowledge base and chat", use_container_width=True):
        for key in ["docs", "chunks", "chat", "vectorizer", "matrix"]:
            st.session_state[key] = [] if key in ["docs", "chunks", "chat"] else None
        st.rerun()

def extract(upload):
    raw = upload.getvalue()
    if upload.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        return [(p.extract_text() or "", i + 1) for i, p in enumerate(reader.pages)]
    return [(raw.decode("utf-8", errors="ignore"), None)]

def make_chunks(documents, size):
    result = []
    for name, text, page in documents:
        words = re.findall(r"\S+", text)
        for start in range(0, len(words), size):
            part = " ".join(words[start:start+size]).strip()
            if len(part) > 30:
                result.append({"text": part, "source": name, "page": page})
    return result

def build_tfidf(chunks):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=25000)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix

def retrieve(question, k):
    qv = st.session_state.vectorizer.transform([question])
    scores = cosine_similarity(qv, st.session_state.matrix).ravel()
    order = scores.argsort()[::-1][:k]
    return [{"score": float(scores[i]), **st.session_state.chunks[i]} for i in order if scores[i] > 0]

def gemini_answer(client, model, question, evidence, history):
    refs = "\n\n".join(
        f"[S{i}] File: {x['source']} | Page: {x['page'] or 'N/A'} | Relevance: {x['score']:.3f}\n{x['text']}"
        for i, x in enumerate(evidence, 1)
    )
    previous = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"""You are a friendly document assistant. Answer naturally and clearly.
Use the retrieved passages as the factual source. Cite claims as [S1], [S2].
If the answer is not in the passages, say so and do not invent it.
Use the previous conversation only to understand follow-up questions.

Conversation:
{previous}

Question: {question}

Retrieved passages:
{refs}

Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or "I couldn't generate a response. Please try again."

tab_chat, tab_files, tab_algorithm = st.tabs(["💬 Chat", "📂 Documents & Analytics", "🧩 Algorithm"])

with tab_files:
    st.subheader("Upload knowledge")
    uploads = st.file_uploader("Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Build beginner RAG index", type="primary", disabled=not uploads):
        extracted = []
        for file in uploads:
            for text, page in extract(file):
                if text.strip():
                    extracted.append((file.name, text, page))
        new_chunks = make_chunks(extracted, chunk_words)
        if new_chunks:
            st.session_state.chunks = new_chunks
            st.session_state.docs = list(dict.fromkeys(x[0] for x in extracted))
            st.session_state.vectorizer, st.session_state.matrix = build_tfidf(new_chunks)
            st.session_state.chat = []
            st.success(f"Indexed {len(new_chunks)} chunks from {len(st.session_state.docs)} documents.")
        else:
            st.error("No readable text found. Scanned PDFs need OCR first.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(st.session_state.docs))
    c2.metric("Text chunks", len(st.session_state.chunks))
    c3.metric("Vocabulary size", len(st.session_state.vectorizer.vocabulary_) if st.session_state.vectorizer else 0)

    if st.session_state.chunks:
        lengths = pd.DataFrame({"Chunk": range(1, len(st.session_state.chunks)+1),
                                "Words": [len(x["text"].split()) for x in st.session_state.chunks],
                                "Source": [x["source"] for x in st.session_state.chunks]})
        st.plotly_chart(px.histogram(lengths, x="Words", color="Source", title="Chunk size distribution"), use_container_width=True)
        st.plotly_chart(px.bar(lengths.groupby("Source", as_index=False).size(), x="Source", y="size", title="Chunks per document"), use_container_width=True)
        with st.expander("Preview indexed chunks"):
            for i, item in enumerate(st.session_state.chunks[:20], 1):
                st.markdown(f"**Chunk {i} · {item['source']} · page {item['page'] or 'N/A'}**")
                st.write(item["text"][:600])

with tab_algorithm:
    st.subheader("Beginner RAG algorithm")
    steps = [
        ("1. Load", "Read PDF/TXT files and extract their text."),
        ("2. Chunk", "Split long documents into smaller word windows."),
        ("3. Vectorize", "TF-IDF converts text into weighted word-feature vectors."),
        ("4. Search", "Cosine similarity compares the question vector with every chunk."),
        ("5. Retrieve", "Select the Top-K most relevant chunks."),
        ("6. Generate", "Gemini writes a conversational answer grounded in retrieved text."),
    ]
    for title, desc in steps:
        st.markdown(f'<div class="panel"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
    st.info("TF-IDF is a lexical retrieval method: it matches important words, not meaning. That limitation motivates the semantic search in Project 2.")

with tab_chat:
    if not st.session_state.chunks:
        st.info("Upload documents and build the index first.")
    elif not api_key:
        st.warning("Add your Gemini API key in the sidebar to enable chat.")
    else:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        question = st.chat_input("Ask something about your documents...")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    client = genai.Client(api_key=api_key)
                    found = retrieve(question, top_k)
                    st.markdown("**Retrieval results**")
                    if found:
                        scores_df = pd.DataFrame([{"Source": f"{x['source']} · p.{x['page'] or '-'}", "Score": x["score"]} for x in found])
                        st.plotly_chart(px.bar(scores_df, x="Score", y="Source", orientation="h", title="Cosine similarity scores"), use_container_width=True)
                        answer = gemini_answer(client, model_name, question, found, st.session_state.chat[:-1])
                    else:
                        answer = "I couldn't find a matching passage in the uploaded documents. Try different wording or upload a relevant document."
                    st.markdown(answer)
                    with st.expander("See retrieved passages"):
                        for i, item in enumerate(found, 1):
                            st.markdown(f"**[S{i}] {item['source']} · page {item['page'] or 'N/A'} · score {item['score']:.3f}**")
                            st.write(item["text"])
                    st.session_state.chat.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Request failed: {e}")

st.caption("Beginner project · TF-IDF RAG · Gemini generation")
