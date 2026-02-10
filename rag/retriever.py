from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "data/chroma_db"

# --- SAME embeddings used in ingestion ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Load vector DB ---
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- Gemini Flash 3 Preview LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",   # Flash preview / latest fast model
    temperature=1.0,
    convert_system_message_to_human=True,
)

# --- Prompt ---
prompt = ChatPromptTemplate.from_template(
    """You are CropSense AI, an expert agriculture advisor helping farmers.

Use the provided context to answer accurately.

Context:
{context}

Question:
{question}

Give clear, practical, step-by-step farming guidance suitable for Indian farmers."""
)

# --- LCEL RAG chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Public function ---
def ask(query: str) -> str:
    return rag_chain.invoke(query)
