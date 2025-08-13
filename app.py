import streamlit as st
from PyPDF2 import PdfReader
import docx
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import logging
import torch
from sentence_transformers import SentenceTransformer

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIR = "db"

# Streamlit page setup
st.set_page_config(page_title="⚖️ AI Legal Chatbot For Lawyers", layout="wide")

# Elegant Dark Theme
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0b1e3f, #102a54) !important; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
h1,h2,h3 { color: #89b4fa !important; }
.model-info { background: rgba(137,180,250,0.1); border-left: 5px solid #89b4fa; padding: 10px; border-radius: 10px; }
.chat-bubble-user { background: rgba(137,180,250,0.25); color: #0b1e3f; padding: 12px; border-radius: 15px 15px 0 15px; margin: 8px 0; max-width: 75%; }
.chat-bubble-bot { background: rgba(22,40,64,0.7); color: #cdd6f4; padding: 12px; border-radius: 15px 15px 15px 0; margin: 8px 0; max-width: 75%; }
.chat-container { max-height: 70vh; overflow-y: auto; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Verify PyTorch installation
try:
    import torch
    logger.info(f"PyTorch version {torch.__version__} detected.")
except ImportError:
    st.error("PyTorch is not installed. Please install it using: pip install torch")
    logger.error("PyTorch is not installed.")
    st.stop()

# File readers
def read_pdf(file):
    try:
        pdf = PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {file.name}")
            st.warning(f"No readable text found in {file.name}. Ensure it's a text-based PDF.")
        else:
            logger.info(f"Successfully read PDF: {file.name}")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {file.name}: {e}")
        st.warning(f"Error reading PDF {file.name}: {e}")
        return ""

def read_docx(file):
    try:
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        if not text.strip():
            logger.warning(f"No text extracted from DOCX: {file.name}")
            st.warning(f"No readable text found in {file.name}.")
        else:
            logger.info(f"Successfully read DOCX: {file.name}")
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX {file.name}: {e}")
        st.warning(f"Error reading DOCX {file.name}: {e}")
        return ""

# Process and store documents
def process_and_store_docs(uploaded_files):
    texts = []
    for file in uploaded_files:
        file_name = file.name.lower()
        if file_name.endswith(".pdf"):
            text = read_pdf(file)
        elif file_name.endswith(".docx"):
            text = read_docx(file)
        elif file_name.endswith(".txt"):
            try:
                text = file.read().decode("utf-8", errors="ignore")
                logger.info(f"Successfully read TXT: {file.name}")
            except Exception as e:
                logger.error(f"Error reading TXT {file.name}: {e}")
                st.warning(f"Error reading TXT {file.name}: {e}")
                text = ""
        else:
            logger.warning(f"Unsupported file format: {file.name}")
            st.warning(f"Unsupported file format: {file.name}")
            text = ""
        if text.strip():
            texts.append(text)

    if not texts:
        logger.error("No readable text found in uploaded files.")
        st.error("No readable text found - upload text-based PDFs/DOCX/TXT")
        return None

    # Split texts into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100
    ).split_text("\n".join(texts))
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        logger.error("No valid chunks created from documents.")
        st.error("No valid chunks created from documents.")
        return None

    # Initialize embeddings with explicit model loading
    try:
        # Load SentenceTransformer model directly to ensure proper device handling
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Successfully initialized HuggingFaceEmbeddings.")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        st.error(f"Failed to initialize embeddings: {e}")
        return None

    # Initialize or update vector database
    try:
        if os.path.exists(PERSIST_DIR):
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            db.add_texts(chunks)
            logger.info("Updated existing Chroma database.")
        else:
            db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
            logger.info("Created new Chroma database.")
        db.persist()
        return db
    except Exception as e:
        logger.error(f"Error creating/storing vector database: {e}")
        st.error(f"Error creating/storing vector database: {e}")
        return None

@st.cache_resource
def load_vector_db():
    if os.path.exists(PERSIST_DIR):
        try:
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Loaded existing Chroma database.")
            return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            st.error(f"Error loading vector database: {e}")
            return None
    return None

# RAG Chain
def create_rag_chain(vectordb):
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        llm = Ollama(model="phi3")
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="You are a legal AI assistant. Use only the provided context to answer. Context:\n{context}\n\nQuestion: {question}\n\nDetailed Answer:"
        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("Successfully created RAG chain.")
        return chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        st.error(f"Error creating RAG chain: {e}")
        return None

# Header
st.markdown("<h1>⚖️ AI LAWYER </h1>", unsafe_allow_html=True)
st.markdown("""
<div class="model-info">
<b>Model:</b> Phi-3 via Ollama (Local, Free)<br>
<b>Embeddings:</b> all-MiniLM-L6-v2 (Local, Free)<br>
<b>Vector DB:</b> Chroma Persistent<br>
<b>Mode:</b> Offline RAG for Legal Q&A
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_files = st.file_uploader("Upload legal documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Clear documents
if st.button("🗑 Clear All Documents"):
    try:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            logger.info("Cleared Chroma database directory.")
        st.session_state.messages = []
        st.success("All stored documents cleared.")
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        st.error(f"Error clearing documents: {e}")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load or create vector database
vectordb = process_and_store_docs(uploaded_files) if uploaded_files else load_vector_db()

if vectordb:
    chain = create_rag_chain(vectordb)
    if chain:
        question = st.chat_input("Ask your legal question...")
        if question:
            with st.spinner("Processing..."):
                try:
                    answer = chain.run(question)
                    st.session_state.messages.append(("user", question))
                    st.session_state.messages.append(("bot", answer))
                    logger.info(f"Processed question: {question}")
                except Exception as e:
                    logger.error(f"Error processing question: {e}")
                    st.error(f"Error processing question: {e}")

        # Display chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for role, msg in st.session_state.messages:
            css_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
            st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Failed to initialize the RAG chain. Please check logs for details.")
else:
    st.info("Please upload at least one readable document to start chatting.")