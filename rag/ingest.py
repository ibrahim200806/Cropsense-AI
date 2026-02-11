import os
import pandas as pd
from PIL import Image
import pytesseract

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------- CONFIG --------
# Define directories for different data types
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
CSV_DIR = os.path.join(DATA_DIR, "raw_csvs")
IMAGE_DIR = os.path.join(DATA_DIR, "raw_images") # New directory for plant images
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

documents = []

# =========================================================
# 1️⃣ LOAD PDFs (Text-based)
# =========================================================
if os.path.exists(PDF_DIR):
    print(f"📂 Scanning PDF directory: {PDF_DIR}")
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(PDF_DIR, file))
                docs = loader.load()
                documents.extend(docs)
                print(f"   - Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")

# =========================================================
# 2️⃣ LOAD CSV FILES (Structured Data)
# =========================================================
if os.path.exists(CSV_DIR):
    print(f"📂 Scanning CSV directory: {CSV_DIR}")
    for file in os.listdir(CSV_DIR):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(CSV_DIR, file))
                # Convert each row into a readable text format
                for _, row in df.iterrows():
                    # Handle potential NaN values
                    clean_row = {k: (v if pd.notna(v) else "N/A") for k, v in row.items()}
                    text = ", ".join([f"{col}: {val}" for col, val in clean_row.items()])
                    
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file, "type": "csv"}
                    ))
                print(f"   - Loaded: {file}")
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")

# =========================================================
# 3️⃣ LOAD IMAGES (OCR for Plant Disease Data)
# =========================================================
# This section was missing in your original code
if os.path.exists(IMAGE_DIR):
    print(f"📂 Scanning Image directory: {IMAGE_DIR}")
    for file in os.listdir(IMAGE_DIR):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            try:
                image_path = os.path.join(IMAGE_DIR, file)
                image = Image.open(image_path)
                
                # Extract text from image using Tesseract OCR
                extracted_text = pytesseract.image_to_string(image)
                
                if extracted_text.strip():
                    documents.append(Document(
                        page_content=extracted_text,
                        metadata={"source": file, "type": "image"}
                    ))
                    print(f"   - OCR Scanned: {file}")
                else:
                    print(f"   ⚠️ No text found in: {file}")
                    
            except Exception as e:
                print(f"   ❌ Error processing image {file}: {e}")

print(f"\n📄 Total documents loaded: {len(documents)}")

if not documents:
    print("⚠️ No documents found! Check your data directories.")
    exit()

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
print("🧠 Generating Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 6️⃣ STORE IN CHROMA VECTOR DB
# =========================================================
print("💾 Saving to Vector Database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print(f"✅ Multimodal ingestion complete! DB stored at: {CHROMA_DIR}")