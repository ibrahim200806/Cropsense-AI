import os
import pandas as pd
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------- CONFIG --------
PDF_DIR = "data/raw_pdfs"
CSV_DIR = "data/raw_pdfs"
CHROMA_DIR = "data/chroma_db"


documents = []

# =========================================================
# 1️⃣ LOAD PDFs
# =========================================================
if os.path.exists(PDF_DIR):
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, file))
            documents.extend(loader.load())

print(f"📄 Loaded PDF pages: {len(documents)}")


# =========================================================
# 2️⃣ LOAD CSV FILES → convert rows to text documents
# =========================================================
csv_docs = []

if os.path.exists(CSV_DIR):
    for file in os.listdir(CSV_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(CSV_DIR, file))

            for _, row in df.iterrows():
                text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                csv_docs.append(Document(page_content=text))

documents.extend(csv_docs)

print(f"📊 Loaded CSV rows as docs: {len(csv_docs)}")


# =========================================================
# 4️⃣ CHUNKING
# =========================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

print(f"🔹 Total chunks created: {len(chunks)}")


# =========================================================
# 5️⃣ EMBEDDINGS
# =========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 6️⃣ STORE IN CHROMA VECTOR DB
# =========================================================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print("✅ Multimodal ingestion complete → Vector DB ready.")
