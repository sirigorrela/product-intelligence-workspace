import os 
import random
import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from ingest import get_database_stats, ingest_data

SUGGESTIONS = [

    "Which feature requests appear most often?",

    "Summarize engineering issues.",

    "What affects enterprise customers?",

    "Which issues remain unresolved?",

    "Which release fixed login timeout?",

    "Summarize product roadmap.",

    "Which team should prioritize next?",

    "What generated the highest negative feedback?"

]

st.set_page_config(page_title="Product Intelligence Agent", page_icon="🧠", layout="wide")
st.title("🧠 Autonomous Product Intelligence System")
st.markdown("Ask complex questions about customer complaints, feature requests, or engineering metrics.")

c1, c2, c3, c4 = st.columns(4)

@st.cache_data
def load_stats():
    return get_database_stats()

stats = load_stats()

DOCUMENT_COUNT = stats["documents"]
CHUNK_COUNT = stats["chunks"]

c1.metric("Documents", DOCUMENT_COUNT)
c2.metric("Chunks", CHUNK_COUNT)
c3.metric("Model", "Gemini")
c4.metric("Vector DB", "Chroma")

with st.sidebar:

    st.title("Workspace")
    
    st.caption("Semantic Search Workspace")

    st.success("🟢 Online")
    
    st.divider()

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose knowledge documents",
        type=["txt", "pdf", "docx", "md"],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, DOCX, Markdown"
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")

        with st.expander("Selected Files", expanded=True):

            for file in uploaded_files:

                st.write(f"📄 {file.name}")
        
    if st.button("📚 Index Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one document before indexing.")
        else:

            os.makedirs("data", exist_ok=True)

            new_files = 0
            replaced_files = 0

            for uploaded_file in uploaded_files:

                file_path = os.path.join("data", uploaded_file.name)

                if os.path.exists(file_path):
                    replaced_files += 1
                else:
                    new_files += 1

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            load_stats.clear()

            with st.spinner("Indexing documents..."):
                stats = ingest_data()


            st.success(
                f"Indexed {stats['documents']} documents into {stats['chunks']} chunks."
            )

            if new_files:
                st.info(f"➕ Added {new_files} new file(s).")

            if replaced_files:
                st.info(f"♻️ Updated {replaced_files} existing file(s).")

            st.rerun()

    st.divider()

    st.subheader("Knowledge Base")

    st.write(f"📄 Documents : {DOCUMENT_COUNT}")
    st.write(f"🧩 Chunks : {CHUNK_COUNT}")
    st.write("🗄️ Vector DB : Chroma")
    st.write("🤖 Model : Gemini 2.5 Flash")

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

# status_box = st.empty()
# status_box.info("🔄 Initializing system and connecting to database...")

try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    from langchain_chroma import Chroma

    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    from langchain_core.tools import Tool
    from langchain_core.messages import HumanMessage

    load_dotenv()

    CHROMA_PATH = "chroma_db"
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            os.environ["GEMINI_API_KEY"] = api_key
        except Exception:
            st.error("GEMINI_API_KEY not found.")
            st.stop()

    @st.cache_resource
    def setup_agent():
        # FIX: Using the active, supported Gemini embedding endpoint
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        if not os.path.exists(CHROMA_PATH):
            st.info("📚 First-time setup. Building knowledge base...")
            ingest_data()
        
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="knowledge_base",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        def search_knowledge(query: str) -> str:
            """Searches company knowledge base."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        rag_tool = Tool(
            name="search_company_knowledge",
            description="Searches and returns excerpts from customer support tickets, PRDs, and meeting notes.",
            func=search_knowledge
        )
        tools = [rag_tool]

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        memory = MemorySaver()

        system_message = """
            You are an AI Product Intelligence Analyst.

            Always use the search_company_knowledge tool before answering.

            Structure every response as:

            ## Summary
            A concise answer.

            ## Evidence
            List the relevant facts from the retrieved documents.

            ## Recommendations
            Provide practical recommendations based only on the retrieved evidence. If the documents do not contain enough information, explicitly say so instead of inventing details.

            Never fabricate information.
            """

        return create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_message,
            checkpointer=memory,
        )

    agent = setup_agent()

    # if agent == "no_db":
    #     st.warning("⚠️ Database folder 'chroma_db' not found. Did you run './venv/bin/python3 ingest.py' first?")
    #     st.stop()

    # status_box.empty()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if len(st.session_state.messages) == 0:

        st.info("👋 Welcome! Select a question to get started.")

        sample_questions = [
            "What are the top customer complaints?",
            "Which feature requests are pending?",
            "Summarize engineering issues.",
            "Which customers are most affected?"
        ]

        cols = st.columns(2)

        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, use_container_width=True):
                    st.session_state.selected_prompt = question
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):

            if message["role"] == "assistant":

                with st.container(border=True):
                    st.markdown(message["content"])

            else:
                st.markdown(message["content"])
            
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    prompt = st.chat_input(
        placeholder=random.choice(SUGGESTIONS)
    )

    if "selected_prompt" in st.session_state:
        prompt = st.session_state.pop("selected_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing product documents..."):
                start = time.time()
                response = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config)
                last_message = response["messages"][-1]

                if isinstance(last_message.content, list):
                    final_answer = "\n".join(
                        block["text"]
                        for block in last_message.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    final_answer = last_message.content
                
                elapsed = time.time() - start
                    
                with st.container(border=True):
                    st.markdown(final_answer)
                    st.caption(f"⏱ Response generated in {elapsed:.2f} sec")
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

except Exception as e:
    # status_box.empty()
    st.error(f"💥 An unexpected error occurred during startup:")
    st.code(str(e))
    

st.divider()

st.caption(
    """
    Product Intelligence Workspace
    Built with Streamlit • LangGraph • Gemini • Chroma
    """
)