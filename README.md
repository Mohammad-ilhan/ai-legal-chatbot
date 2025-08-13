# ⚖️ AI Legal Chatbot for lawyers

An AI-powered chatbot for answering legal questions using **Retrieval-Augmented Generation (RAG)** with a local model.  
You can upload PDF, DOCX, or TXT documents, and the bot will search and answer based only on the uploaded context.

## 🚀 Features
- Upload and process multiple PDF/DOCX/TXT legal documents
- Persistent vector database using **ChromaDB**
- Local embeddings with **Sentence Transformers**
- Local LLM with **Ollama** (Phi-3 model)
- Elegant dark mode Streamlit UI
- Chat history for ongoing conversation


## **2️⃣ Create `requirements.txt`**
```txt
streamlit
PyPDF2
python-docx
langchain
langchain-community
chromadb
sentence-transformers
torch

