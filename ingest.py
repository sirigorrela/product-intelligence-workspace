import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
# DATA_PATH = "mock.data" if os.path.exists("mock.data") else "mock_data"
DATA_PATH = "mock_data"

def ingest_data():
    print("🚀 Starting database creation with Google Gemini...")
    os.makedirs(DATA_PATH, exist_ok=True)
    
    sample_file = os.path.join(DATA_PATH, "data.txt")
    print(f"📝 Writing rich SaaS data into '{sample_file}'...")
    with open(sample_file, "w") as f:
        f.write("""
        CUSTOMER SUPPORT REPORT - Q2
        The most common customer complaints this quarter are regarding slow load times on the analytics dashboard and the lack of a dark mode feature. The slow load times are currently unresolved and are affecting our enterprise tier customers the most.
        
        PRODUCT ROADMAP NOTES
        Feature requests appearing most frequently across tickets are: 1. Dark Mode, 2. Export to PDF, 3. Slack Integration. Dark mode and PDF export have not yet been prioritized by the engineering team. The analytics product area generates the highest volume of negative feedback.
        
        ENGINEERING METRICS
        The login timeout bug reported by 40 customers last month was eventually fixed in release v2.4. Improving the database indexing had the highest impact on customer satisfaction, reducing load times for standard users by 40%.
        """)

    print(f"📂 Loading documents from '{DATA_PATH}'...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("Error: No .txt files found!")
        return

    print(f"📄 Loaded {len(documents)} document(s). Splitting text...")
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
    )

    print("SUCCESS! Database created and saved to 'chroma_db' folder!")

if __name__ == "__main__":
    ingest_data()