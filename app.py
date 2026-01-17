from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    files = list(WATCH_PATH.glob("*.pdf"))

    if not files:
        return "No Files in ./knowledge_base folder"
    
    total_chunks = 0

    for file in files:
        try:
            if file.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file))

            docs = loader.load()
            if not docs: 
                continue
        
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap = 100
            )

            chunks = splitter.split_documents(docs)

            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)

        except Exception as e:
            return f"Error Processing {file.name}: {str(e)}"
    
    return f"Index {len(files)} files -> {total_chunks} chunks"

def query_rag(question : str):
    """Retrieves from the DB to generate answers"""

    docs = vectorstore.similarity_search(question, k=5)

    if not docs:
        return "No documents found for your question"
    
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt_template = f"""
    You are an agent that summarizes data that answers a question given a context.

    Given this context: {context}

    Make a summary that answers this question: {question}
    """

    response = llm.invoke(prompt_template)
    return response.content.strip()
