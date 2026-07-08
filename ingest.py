import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
# DATA_PATH = "mock.data" if os.path.exists("mock.data") else "mock_data"
# DATA_PATH = "data"

def get_database_stats(data_path="data"):
    """
    Returns the number of supported documents and chunks.
    """

    document_count = 0
    chunk_count = 0

    documents = load_documents(data_path)
    document_count = len(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    chunk_count = len(chunks)

    return {
        "documents": document_count,
        "chunks": chunk_count,
    }

def load_documents(data_path):
    documents = []

    for file in sorted(os.listdir(data_path)):
        
        if file.startswith("."):
            continue

        file_path = os.path.join(data_path, file)

        if file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path)

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)

        else:
            print(f"Skipping unsupported file: {file}")
            continue

        try:
            documents.extend(loader.load())
            print(f"📄 Loaded: {file}")

        except Exception as e:
            print(f"❌ Failed to load {file}")
            print(e)

    return documents

def ingest_data(data_path="data"):
    print("🚀 Starting database creation with Google Gemini...")
    os.makedirs(data_path, exist_ok=True)
    
    # sample_file = os.path.join(DATA_PATH, "data.txt")
    # print(f"📝 Writing rich SaaS data into '{sample_file}'...")
    # with open(sample_file, "w") as f:
    #     f.write("""
    #     CUSTOMER SUPPORT REPORT - Q2
    #     The most common customer complaints this quarter are regarding slow load times on the analytics dashboard and the lack of a dark mode feature. The slow load times are currently unresolved and are affecting our enterprise tier customers the most.
        
    #     PRODUCT ROADMAP NOTES
    #     Feature requests appearing most frequently across tickets are: 1. Dark Mode, 2. Export to PDF, 3. Slack Integration. Dark mode and PDF export have not yet been prioritized by the engineering team. The analytics product area generates the highest volume of negative feedback.
        
    #     ENGINEERING METRICS
    #     The login timeout bug reported by 40 customers last month was eventually fixed in release v2.4. Improving the database indexing had the highest impact on customer satisfaction, reducing load times for standard users by 40%.
    #     """)

    print(f"📂 Loading documents from '{data_path}'...")
    # loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    # documents = loader.load()
    # if os.path.exists(CHROMA_PATH):
    #     print("Removing old database...")
    #     shutil.rmtree(CHROMA_PATH)
        
    documents = load_documents(data_path)

    if not documents:
        print("No supported documents were found.")
        return

    print(f"📄 Indexed {len(documents)} document(s). Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    if not chunks or len(chunks) == 0:
        print("Error: Documents were empty! 0 chunks created.")
        return

    print(f"🧩 Split into {len(chunks)} chunks. Building Vector Database with Gemini...")
    
    # FIX: Using the active, supported Gemini embedding endpoint
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="knowledge_base",
    )

    print("SUCCESS! Database created and saved to 'chroma_db' folder!")

    return {
        "documents": len(documents),
        "chunks": len(chunks)
    }

if __name__ == "__main__":
    ingest_data()