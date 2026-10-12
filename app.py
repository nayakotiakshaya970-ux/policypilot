import os
import time
import uuid

import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

load_dotenv()

MODEL = "gemini-3.8-flash"   # model name from Google's error message; check AI Studio if it changes
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_QUESTIONS = 20           # per session, protects your free-tier limits
NOT_FOUND = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = f"""You are PolicyPilot, an assistant that answers questions about company documents.
Rules:
1. Use ONLY the provided sources. Never use outside knowledge.
2. After each fact, cite the source like (filename, p.X).
3. If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
4. Keep answers short and clear."""


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]  # used when deployed
    except Exception:
        return None


def chunk_text(text):
    """Cut text into overlapping pieces so no sentence is lost at a boundary."""
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE].strip()
        if piece:
            chunks.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def new_collection():
    client = chromadb.EphemeralClient()
    return client.create_collection(f"docs_{uuid.uuid4().hex}")


def index_files(files, collection):
    """Read every page, chunk it, and store it with its file name and page number."""
    ids, docs, metas = [], [], []
    for f in files:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for j, chunk in enumerate(chunk_text(text)):
                ids.append(f"{f.name}-p{page_num}-c{j}")
                docs.append(chunk)
                metas.append({"source": f.name, "page": page_num})
    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def call_model(client, contents):
    """Call Gemini, retrying when Google is busy (503) or the free-tier limit is hit (429)."""
    delay = 3
    for attempt in range(5):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2000,
                ),
            )
        except errors.ServerError:
            if attempt == 4:
                raise
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429 or attempt == 4:
                raise
        time.sleep(delay)
        delay *= 2


def answer_question(client, collection, question, prior_chat):
    start = time.time()

    # Add the previous question so follow-ups like "what about contractors?" still search well
    previous = [m["content"] for m in prior_chat if m["role"] == "user"]
    search_text = f"{previous[-1]} {question}" if previous else question

    found = collection.query(
        query_texts=[search_text],
        n_results=min(TOP_K, collection.count()),
    )
    sources = [
        {"file": m["source"], "page": m["page"], "text": d}
        for d, m in zip(found["documents"][0], found["metadatas"][0])
    ]
    context = "\n\n".join(
        f"[Source {i}: {s['file']}, page {s['page']}]\n{s['text']}"
        for i, s in enumerate(sources, start=1)
    )

    # Gemini uses the role "model" where other APIs use "assistant"
    contents = []
    for m in prior_chat[-6:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=f"Sources:\n{context}\n\nQuestion: {question}")],
    ))

    response = call_model(client, contents)
    usage = response.usage_metadata
    return {
        "answer": response.text or NOT_FOUND,
        "sources": sources,
        "tokens": (usage.total_token_count or 0) if usage else 0,
        "seconds": round(time.time() - start, 1),
    }


# ---------------- Page ----------------
st.set_page_config(page_title="PolicyPilot", layout="wide")
st.title("🧭 PolicyPilot")
st.caption("Ask questions about your company documents. Every answer cites its source page.")

api_key = get_api_key()
if not api_key:
    st.error("No API key found. Add GEMINI_API_KEY to your .env file.")
    st.stop()
client = genai.Client(api_key=api_key)

for key, default in [("collection", None), ("indexed_key", None), ("chat", []), ("tokens", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Clear chat"):
        st.session_state.chat = []
    st.metric("Tokens used this session", st.session_state.tokens)

if files:
    file_key = tuple((f.name, f.size) for f in files)
    if st.session_state.indexed_key != file_key:
        with st.spinner("Reading and indexing documents (first run downloads a small model)..."):
            collection = new_collection()
            n_chunks = index_files(files, collection)
        st.session_state.collection = collection
        st.session_state.indexed_key = file_key
        st.session_state.chat = []
        if n_chunks == 0:
            st.warning("No text found. These may be scanned images, which need OCR.")
            st.session_state.collection = None
        else:
            st.sidebar.success(f"Indexed {n_chunks} passages from {len(files)} file(s)")

ready = st.session_state.collection is not None and st.session_state.collection.count() > 0

tab_chat, tab_eval = st.tabs(["💬 Ask", "✅ Test accuracy"])

with tab_chat:
    if not ready:
        st.info("Upload at least one PDF in the sidebar to begin.")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m["role"] == "assistant" and m.get("sources"):
                    with st.expander("Sources used"):
                        for s in m["sources"]:
                            st.markdown(f"**{s['file']} — page {s['page']}**")
                            st.caption(s["text"][:400] + "...")

        asked = sum(1 for m in st.session_state.chat if m["role"] == "user")
        question = st.chat_input("Ask a question about your documents")
        if question:
            if asked >= MAX_QUESTIONS:
                st.warning("Question limit reached for this session. Click 'Clear chat' to reset.")
            else:
                result = None
                try:
                    with st.spinner("Searching documents..."):
                        result = answer_question(
                            client, st.session_state.collection, question, st.session_state.chat
                        )
                except Exception as e:
                    st.error("The AI service could not answer this question. "
                             "Please wait a minute and ask again.")
                    st.code(f"{type(e).__name__}: {str(e)[:500]}")
                if result:
                    st.session_state.chat.append({"role": "user", "content": question})
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                    st.session_state.tokens += result["tokens"]
                    st.rerun()

with tab_eval:
    st.write("Upload a CSV with two columns: **question** and **expected** (a word or phrase "
             "the correct answer must contain).")
    st.caption("For a question that isn't in your documents, set expected to: couldn't find")
    test_file = st.file_uploader("Test CSV", type="csv", key="eval_file")
    if test_file and not ready:
        st.warning("Upload your PDFs in the sidebar first.")
    if test_file and ready and st.button("Run accuracy test"):
        tests = pd.read_csv(test_file)
        rows = []
        progress = st.progress(0)
        for i, row in tests.iterrows():
            result = answer_question(client, st.session_state.collection, str(row["question"]), [])
            passed = str(row["expected"]).lower() in result["answer"].lower()
            rows.append({
                "question": row["question"],
                "expected": row["expected"],
                "answer": result["answer"],
                "passed": passed,
            })
            st.session_state.tokens += result["tokens"]
            progress.progress((i + 1) / len(tests))
            time.sleep(2)  # small pause to stay under the free-tier rate limit
        report = pd.DataFrame(rows)
        st.metric("Accuracy", f"{report['passed'].mean():.0%}")
        st.dataframe(report, width="stretch")
        st.download_button("📥 Download results", report.to_csv(index=False), "policypilot_eval.csv")
    