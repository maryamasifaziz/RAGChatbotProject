# 📝 RAG Q&A Chatbot — Multi-PDF Question Answering

> **An intelligent chatbot that answers questions from multiple PDF documents using Retrieval-Augmented Generation (RAG) with persistent chat history, query rewriting, and strict hallucination prevention — powered by LangChain, ChromaDB, Groq, and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)

---

## 🚀 Live Demo
🔗 [View Live App](https://ragchatbotproject12.streamlit.app/)

---

## 🧠 Overview
Upload multiple PDFs and ask questions in natural language. The chatbot uses a **two-stage RAG pipeline** — first rewriting your query using chat history for better retrieval, then generating grounded answers using **Groq's LLaMA-3.1-8b** model. Answers are strictly limited to document content — no hallucinations, no outside knowledge.

---

## ✨ Features
- Upload and query **multiple PDFs simultaneously**
- **Query rewriting** — resolves pronouns and follow-ups for accurate retrieval
- **Persistent chat history** — saved to disk per session and document
- **Strict hallucination prevention** — refuses to answer outside document scope
- **MMR retrieval** — diverse, non-redundant chunk selection
- HuggingFace embeddings (`all-MiniLM-L6-v2`) cached in session for performance
- In-memory ChromaDB vectorstore — no stale chunks between uploads
- Debug panel showing rewritten query and retrieved chunks
- Session ID support for multiple independent conversations

---

## 🏗️ Architecture
```
PDF Upload (Multiple Files)
        ↓
PyPDFLoader → RecursiveCharacterTextSplitter
(chunk_size=1200, overlap=120)
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
ChromaDB In-Memory Vectorstore
        ↓
User Query + Chat History
        ↓
Query Rewriting (Groq LLaMA-3.1)
        ↓
MMR Retrieval (k=5, fetch_k=20)
        ↓
Groq LLaMA-3.1-8b → Grounded Answer
        ↓
Streamlit Chat Interface + Persistent Memory
```

---

## 🧰 Tech Stack
| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline orchestration |
| Groq (LLaMA-3.1-8b-instant) | LLM for query rewriting + answer generation |
| ChromaDB | In-memory vector store |
| HuggingFace Embeddings | Sentence embeddings (all-MiniLM-L6-v2) |
| PyPDFLoader | PDF text extraction |
| Streamlit | Chat interface + deployment |

---

## 📁 Project Structure
```
RAGChatbotProject/
├── RAG_Chatbot.py      # Full RAG pipeline + Streamlit UI
├── chat_memory/        # Persistent chat history (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally
```bash
git clone https://github.com/maryamasifaziz/RAGChatbotProject
cd RAGChatbotProject
pip install -r requirements.txt
streamlit run RAG_Chatbot.py
```

Add your Groq API key in `.env`:
```
GROQ_API_KEY=your_groq_api_key
```
Or enter it directly in the app sidebar.

---

## 🔑 Get Your Free Groq API Key
1. Go to **console.groq.com**
2. Sign up free
3. Create an API key
4. Paste it in the sidebar or `.env`

---

## 🛡️ Hallucination Prevention
The chatbot strictly answers **only from uploaded documents**. If the answer isn't in the PDFs, it responds with:
> *"Out of scope - not found in provided documents."*

Even if the user pushes back, it maintains this boundary — no outside knowledge is used under any circumstance.

---

## 👤 Author
**Maryam Asif**  
🎓 FAST NUCES  
🔗 [LinkedIn](https://linkedin.com/maryamasifaziz) | [GitHub](https://github.com/maryamasifaziz)
