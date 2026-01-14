import streamlit as st
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG ---
WATCH_PATH = Path("./knowledge_base")
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text:latest"

