from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from pathlib import Path
import streamlit as st

# --- CONFIG ---
WATCH_PATH = Path("./knowledge_base")
DB_PATH = "./chroma_db"
MODEL_NAME = "qwen3:4b"
EMBED_MODEL = "nomic-embed-text:latest"

WATCH_PATH.mkdir(exist_ok=True)

@st.cache_resource
def initialize_llm():
    """Initializes the LLM"""
    llm = ChatOllama(model=MODEL_NAME, temperature = 0.7)
    
    return llm

@st.cache_resource
def initialize_vectorstore():
    """Initializes the Vectorstore"""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory = str(DB_PATH),
        embedding_finction = embeddings,
        collection_name = "documents"
    )

    return vectorstore

llm = initialize_llm()
vectorstore = initialize_vectorstore()

def index_folder():
    """Index the files for the student handbook"""
    pdf_files = list(WATCH_PATH.glob("*.pdf"))
