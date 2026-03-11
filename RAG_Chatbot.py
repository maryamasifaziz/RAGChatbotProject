# import (python built-ins)
import os
import json
import hashlib
import tempfile
import streamlit as st
from dotenv import load_dotenv

## imports langchain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # fix 1: updated import
from langchain_chroma import Chroma

# setup
load_dotenv()
st.set_page_config(page_title="RAG Q&A ", layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Chat History")

# Persistent memory helpers
MEMORY_DIR = "chat_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def _memory_path(key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return os.path.join(MEMORY_DIR, f"{safe}.json")

def load_history_from_disk(key: str) -> ChatMessageHistory:
    history = ChatMessageHistory()
    path = _memory_path(key)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                if rec["role"] == "human":
                    history.add_user_message(rec["content"])
                elif rec["role"] == "ai":
                    history.add_ai_message(rec["content"])
        except (json.JSONDecodeError, KeyError):
            pass
    return history

def save_history_to_disk(key: str, history: ChatMessageHistory) -> None:
    records = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            records.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            records.append({"role": "ai", "content": msg.content})
    with open(_memory_path(key), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# Sidebar
with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs -> Ask questions -> Get Answers")

# Fix 2: read from st.secrets for Streamlit Cloud, fallback to .env for local
def _get_api_key() -> str:
    if api_key_input:
        return api_key_input.strip()
    try:
        return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        pass
    val = os.getenv("GROQ_API_KEY")
    return val.strip() if val else ""

api_key = _get_api_key()
if not api_key:
    st.warning("Please enter your Groq API Key (or set GROQ_API_KEY in Streamlit secrets / .env)")
    st.stop()

# Fix 3: embeddings cached in session_state so model loads only once
# (avoids re-downloading on every Streamlit rerun)
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
embeddings = st.session_state.embeddings

# File upload
uploaded_files = st.file_uploader(
    " 📚 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin")
    st.stop()

# Doc key: MD5 of actual file contents via getvalue() — always readable,
# unlike f.read() which returns empty bytes after first access on Streamlit.
# Scopes both vectorstore and chat history to the exact uploaded documents.
doc_key = hashlib.md5(
    str(tuple(sorted(
        (f.name, hashlib.md5(f.getvalue()).hexdigest())
        for f in uploaded_files
    ))).encode()
).hexdigest()[:12]

# Build vectorstore only when doc_key changes (new files uploaded)
if st.session_state.get("doc_key") != doc_key:

    all_docs = []
    tmp_paths = []

    for pdf in uploaded_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = pdf.name
        all_docs.extend(docs)

    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    splits = text_splitter.split_documents(all_docs)

    # Fix 4: in-memory vectorstore — no persist_directory so no disk files,
    # no Windows file locking, and no stale chunks from previous uploads.
    vectorstore = Chroma.from_documents(splits, embeddings)

    st.session_state.doc_key = doc_key
    st.session_state.vectorstore = vectorstore
    st.session_state.indexed_chunks = len(splits)
    st.session_state.num_pages = len(all_docs)
    st.session_state.chathistory = {}  # clear history cache on new doc

retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.success(f"✅ Loaded {st.session_state.num_pages} pages from {len(uploaded_files)} PDFs")
st.sidebar.write(f"🔍 Indexed {st.session_state.indexed_chunks} chunks for retrieval")
st.sidebar.write(f"💾 Memory: `{os.path.abspath(MEMORY_DIR)}`")

# Helper
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimizer for a document retrieval system.\n\n"
     "Your job: given the conversation history and the user's latest message, produce ONE "
     "precise, self-contained search query suitable for semantic search over documents.\n\n"
     "Rules:\n"
     "- Resolve ALL pronouns and references ('it', 'this', 'that approach', 'the algorithm', "
     "'the previous one') by replacing them with the actual noun/concept from the chat history.\n"
     "- If the user is asking a follow-up (e.g. 'explain more', 'give an example', 'why?'), "
     "expand it into a full question using the topic from the previous exchange.\n"
     "- If the user asks for a table, chart, or summary of financial data (e.g. 'give me a table', "
     "'show financials', 'revenue table'), rewrite it as a specific data retrieval query such as "
     "'revenue net income operating profit financial figures' so the right chunks are retrieved.\n"
     "- Keep the query focused and specific — avoid vague terms.\n"
     "- Output ONLY the rewritten query. No explanation, no punctuation at the end."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert assistant that answers questions using ONLY the provided document context.\n\n"
     "ANSWER QUALITY RULES:\n"
     "1. Structure your answer clearly — use headings, bullet points, or numbered steps when the "
     "topic has multiple parts or a sequence.\n"
     "1a. NEVER fabricate, invent, estimate, or assume ANY information — this applies to ALL "
     "output formats: tables, lists, summaries, examples, charts, timelines, or any other format.\n"
     "Every single fact, number, name, and data point in your answer MUST come directly from "
     "the provided context. If the context does not contain enough information, say so explicitly "
     "instead of filling gaps with guesses or general knowledge.\n"
     "2. Be thorough and detailed. Do not give one-line answers. Explain the 'why' and 'how', "
     "not just the 'what'.\n"
     "3. Use examples, figures, or data from the context to support your answer when available.\n"
     "4. For follow-up questions, build on the previous answer — do not repeat what was already "
     "explained unless the user asks for a recap.\n\n"
     "SCOPE RULES:\n"
     "5. Use ONLY the provided context. Do NOT use outside knowledge under ANY circumstances.\n"
     "6. If the context partially covers the question, answer only what is covered.\n"
     "7. If the question cannot be answered using the provided context, reply EXACTLY with\n"
     "this phrase and nothing else: 'Out of scope - not found in provided documents.'\n"
     "8. Even if the user says 'but tell me', 'tell me anyway', 'use your knowledge', or\n"
     "pushes back in any way — still reply ONLY with: 'Out of scope - not found in provided documents.'\n"
     "NEVER mention general knowledge. NEVER offer to answer from memory. NEVER explain what you know.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Chat history scoped per session + document
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str) -> ChatMessageHistory:
    scoped_key = f"{session_id}__{doc_key}"
    if scoped_key not in st.session_state.chathistory:
        st.session_state.chathistory[scoped_key] = load_history_from_disk(scoped_key)
    return st.session_state.chathistory[scoped_key]

# Chat UI
session_id = st.text_input(" 🆔 Session ID ", value="default_session")
user_q = st.chat_input("💬 Ask a question...")

if user_q:
    scoped_key = f"{session_id}__{doc_key}"
    history = get_history(session_id)

    # Fix 5: LLM initialized here with fresh api_key on every question,
    # so key changes in sidebar are always picked up immediately.
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    # If the last assistant reply was out-of-scope and user is pushing back
    # (e.g. "but tell me", "just answer", "tell me anyway"), refuse again.
    REFUSAL = "Out of scope - not found in provided documents."
    last_messages = history.messages
    if last_messages:
        last_ai = next(
            (m.content for m in reversed(last_messages) if isinstance(m, AIMessage)),
            ""
        )
        pushy_phrases = [
            "but tell me", "tell me anyway", "just tell me", "just answer",
            "answer anyway", "use your knowledge", "from your knowledge",
            "you know it", "i know you know", "pretend", "ignore", "forget"
        ]
        if REFUSAL.lower() in last_ai.lower() and any(
            p in user_q.lower() for p in pushy_phrases
        ):
            st.chat_message("user").write(user_q)
            st.chat_message("assistant").write(REFUSAL)
            history.add_user_message(user_q)
            history.add_ai_message(REFUSAL)
            save_history_to_disk(scoped_key, history)
            st.stop()

    # 1) Rewrite question with history
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )

    try:
        standalone_q = llm.invoke(rewrite_msgs).content.strip()
    except Exception as e:
        st.error(f"❌ LLM error: {e}\n\nCheck your Groq API key is valid.")
        st.stop()

    # 2) Retrieve chunks
    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of scope — not found in provided documents."
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        save_history_to_disk(scoped_key, history)
        st.stop()

    # 3) Build context string
    context_str = _join_docs(docs)

    # 4) Generate answer
    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str
    )

    try:
        answer = llm.invoke(qa_msgs).content
    except Exception as e:
        st.error(f"❌ LLM error: {e}\n\nCheck your Groq API key is valid.")
        st.stop()

    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)
    save_history_to_disk(scoped_key, history)

    # Debug panels
    with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📑 Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
