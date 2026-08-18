import streamlit as st
import os
from dotenv import load_dotenv

from src.document_loader import load_document
from src.text_cleaner import clean_text
from src.contract_parser import extract_metadata
from src.chunking import chunk_document
from src.embeddings import get_embeddings_model
from src.vector_store import load_vector_store, add_to_vector_store, save_vector_store, get_indexed_documents
from src.retriever import retrieve_context
from src.rag_service import generate_answer

# Load environment variables (.env)
load_dotenv()

st.set_page_config(page_title="Contract RAG Analyzer", layout="wide")

# Inject Custom CSS for Vintage/Classic Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@400;600;700&family=Lora:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Lora', serif !important;
    color: #3a403d !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif !important;
    color: #2c3330 !important;
    font-weight: 500 !important;
}

/* Main Background */
.stApp {
    background-color: #FAF7F2;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F4EFE6 !important;
    border-right: 1px solid #E6DFD3 !important;
}

/* Primary Button */
button[kind="primary"] {
    background-color: #93C572 !important;
    color: #1A3022 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.5rem 1rem !important;
    font-family: 'Lora', serif !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background-color: #7DB05E !important;
}

/* Secondary Button */
button[kind="secondary"] {
    background-color: #DCE8D1 !important;
    border: 1px solid #93C572 !important;
    color: #1A3022 !important;
    border-radius: 4px !important;
    font-family: 'Lora', serif !important;
    font-weight: 500 !important;
}
button[kind="secondary"]:hover {
    background-color: #93C572 !important;
}

/* Inputs & Textareas */
.stTextInput > div > div > input, 
.stSelectbox > div > div, 
.stTextArea > div > div > textarea {
    background-color: #F0ECE1 !important;
    border: 1px solid #DED9CD !important;
    border-radius: 4px !important;
    color: #6D6964 !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextInput > div > div > input:focus, 
.stSelectbox > div > div:focus, 
.stTextArea > div > div > textarea:focus {
    box-shadow: 0 0 0 1px #DED9CD !important;
    border-color: #7BA393 !important;
}

::placeholder {
    color: #A19C95 !important;
}

/* Expander/Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FAF7F2 !important;
    border: 1px solid #E6DFD3 !important;
    border-radius: 6px !important;
}

/* File Uploader */
[data-testid="stFileUploadDropzone"] {
    background-color: #F8F5F0 !important;
    border: 1px dashed #C9C4B9 !important;
    border-radius: 4px !important;
}

/* Topbar (Pista Green) */
[data-testid="stHeader"] {
    background-color: #DCE8D1 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_embeddings():
    return get_embeddings_model()

embeddings_model = init_embeddings()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store(embeddings_model)

if "chat_sessions" not in st.session_state:
    from src.session_store import load_sessions
    st.session_state.chat_sessions = load_sessions()

st.title("📄 Contract Search & Analysis Using RAG")
st.markdown("Upload contracts (PDF/DOCX) and ask questions to analyze legal clauses and risks.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Contracts")
    uploaded_files = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Process Contracts"):
        if not uploaded_files:
            st.warning("Please upload at least one contract.")
        else:
            with st.spinner("Processing..."):
                all_chunks = []
                
                # Initialize PostgreSQL Database
                try:
                    from src.db_models import init_db
                    init_db()
                except Exception as e:
                    st.error(f"Failed to initialize PostgreSQL: {e}")
                    
                for file in uploaded_files:
                    file_bytes = file.getbuffer()
                    
                    try:
                        # 1. Postgres Deduplication via File Hash
                        from src.db_service import get_file_hash, is_duplicate_file, save_contract_metadata
                        file_hash = get_file_hash(file_bytes)
                        
                        if is_duplicate_file(file_hash):
                            st.warning(f"Exact duplicate detected for {file.name}. Skipping.")
                            continue
                            
                        # Prevent name-based duplicates in ES as fallback
                        if st.session_state.vector_store:
                            indexed = get_indexed_documents(st.session_state.vector_store)
                            if file.name in indexed:
                                st.warning(f"{file.name} is already indexed by name. Skipping.")
                                continue

                        # Save temporarily to load
                        temp_path = os.path.join("data", "contracts", file.name)
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)
                            
                        # 2. Load
                        raw_text = load_document(temp_path)
                        # 3. Clean
                        cleaned_text = clean_text(raw_text)
                        # 4. Parse Metadata
                        metadata = extract_metadata(cleaned_text, file.name)
                        
                        # 5. Save to Postgres
                        save_contract_metadata(metadata, file_hash, file.name)
                        
                        # 6. Chunk
                        chunks = chunk_document(cleaned_text, metadata)
                        all_chunks.extend(chunks)
                        st.success(f"Processed: {file.name}")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                
                # 5. Embed and Store
                if all_chunks:
                    st.session_state.vector_store = add_to_vector_store(
                        all_chunks, 
                        embeddings_model, 
                        st.session_state.vector_store
                    )
                    save_vector_store(st.session_state.vector_store)
                    st.success("Indexing complete!")
                    
    st.divider()
    st.header("Database")
    if st.session_state.vector_store:
        indexed_files = get_indexed_documents(st.session_state.vector_store)
        st.write(f"Indexed Contracts: {len(indexed_files)}")
        with st.expander("View Indexed Files"):
            for f in indexed_files:
                st.write(f"- {f}")
    else:
        st.write("No contracts indexed yet.")
        
    if st.button("Clear Database"):
        st.session_state.confirm_clear = True
        
    if st.session_state.get("confirm_clear"):
        with st.container(border=True):
            st.warning("⚠️ Are you sure you want to clear the entire database? This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Clear It", type="primary"):
                    from src.vector_store import clear_vector_store
                    from src.db_service import clear_postgres_db
                    clear_vector_store()
                    clear_postgres_db()
                    st.session_state.vector_store = None
                    st.session_state.confirm_clear = False
                    st.session_state.chat_sessions = []
                    from src.session_store import save_sessions
                    save_sessions([])
                    st.session_state.qa_history = []
                    st.session_state.current_session_id = None
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()
                    
    history_container = st.container()

# Main QA Interface
st.header("Ask Questions")

if st.session_state.vector_store:
    indexed_files = get_indexed_documents(st.session_state.vector_store)
    options = ["All Contracts"] + indexed_files
    selected_resume = st.selectbox("Target Contract", options)
else:
    selected_resume = "All Contracts"

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

query_input = st.text_input("Enter your query (e.g., 'What is the limitation of liability?', 'Find the auto-renewal clause')", key="main_query_input")

if st.button("Search"):
    if not query_input:
        st.warning("Please enter a query.")
    elif not st.session_state.vector_store:
        st.warning("Please upload and process some contracts first.")
    else:
        import uuid
        new_session_id = str(uuid.uuid4())
        
        st.session_state.qa_history = []
        import re
        raw_queries = re.split(r'(?<=[.?!])\s+|\n+', query_input)
        queries = [q.strip() for q in raw_queries if q.strip()]
        for q in queries:
            st.session_state.qa_history.append({"chain": [{"q": q, "result": None, "query_type": None}]})
            
        title = queries[0] if queries else "Empty Search"
        
        st.session_state.chat_sessions.append({
            "id": new_session_id,
            "title": title,
            "qa_history": st.session_state.qa_history
        })
        st.session_state.current_session_id = new_session_id
        from src.session_store import save_sessions
        save_sessions(st.session_state.chat_sessions)

if st.session_state.qa_history and st.session_state.vector_store:
    from src.query_parser import parse_query
    from src.retriever import retrieve_context
    from src.rag_service import generate_answer, get_llm, contextualize_query
    
    # Helper to render the results
    def render_result_ui(q, query_type, result):
        if query_type == "error":
            st.error(f"Failed to generate answer: {result.get('error')}")
            return
            
        st.info(f"Query routed to: **{query_type.replace('_', ' ').title()}**")
        st.markdown("#### Answer")
        st.write(result.get("answer", ""))
        
        if result.get("sources"):
            st.markdown("#### Sources")
            for source in result.get("sources", []):
                st.write(f"- {source['file']}")
                
    num_indexed = len(get_indexed_documents(st.session_state.vector_store)) if st.session_state.vector_store else 1

    # Render Sessions
    for session_idx, session in enumerate(st.session_state.qa_history):
        for node_idx, node in enumerate(session["chain"]):
            q = node["q"]
            
            # Use smaller headers for follow-ups
            if node_idx == 0:
                st.markdown(f"### Q: {q}")
            else:
                st.markdown(f"#### ↳ Follow-up: {q}")
            
            if node["result"] is None:
                with st.spinner(f"Analyzing: {q}..."):
                    try:
                        llm = get_llm()
                        
                        search_q = q
                        filter_dict = {"source_file": selected_resume} if selected_resume != "All Contracts" else None
                        
                        if node_idx > 0:
                            history = []
                            for prev_node in session["chain"][:node_idx]:
                                res = prev_node.get("result") or {}
                                ai_msg = res.get("answer", "")
                                history.append({"q": prev_node["q"], "a": ai_msg})
                            search_q = contextualize_query(q, history)
                            st.caption(f"*(Interpreted as: {search_q})*")
                            
                        parsed_query = parse_query(search_q, llm)
                        query_type = parsed_query["query_type"]
                        
                        # Just do standard RAG retrieval for all contract queries
                        retrieved_docs = retrieve_context(search_q, st.session_state.vector_store, k=15, filter_dict=filter_dict)
                        answer = generate_answer(search_q, retrieved_docs)
                        
                        # Keep track of sources to pass to follow-ups
                        sources = [{"file": doc.metadata.get("source_file", "Unknown")} for doc in retrieved_docs if doc.metadata.get("source_file")]
                        # Deduplicate sources
                        unique_sources = []
                        seen_files = set()
                        for s in sources:
                            if s["file"] not in seen_files:
                                seen_files.add(s["file"])
                                unique_sources.append(s)
                                
                        result = {"answer": answer, "sources": unique_sources}
                        
                        node["result"] = result
                        node["query_type"] = query_type
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        node["result"] = {"error": str(e)}
                        node["query_type"] = "error"
                        
            from src.session_store import save_sessions
            save_sessions(st.session_state.chat_sessions)
            render_result_ui(q, node["query_type"], node["result"])
            
        # UI block for follow up styling
        with st.container(border=True):
            st.markdown("<div style='font-family: Inter, sans-serif; color: #6D6964; margin-bottom: 8px;'>💬 Ask a follow-up about this answer</div>", unsafe_allow_html=True)
            col1, col2 = st.columns([5, 1])
            with col1:
                fu_q = st.text_input("Any query in your mind?", key=f"fu_input_{session_idx}_{len(session['chain'])}", label_visibility="collapsed", placeholder="Any query in your mind?")
            with col2:
                if st.button("🔍 Ask", key=f"fu_btn_{session_idx}_{len(session['chain'])}", use_container_width=True):
                    if fu_q:
                        st.session_state.qa_history[session_idx]["chain"].append({"q": fu_q, "result": None, "query_type": None})
                        from src.session_store import save_sessions
                        save_sessions(st.session_state.chat_sessions)
                        st.rerun()
                        
        st.divider()

# Render Search History in Sidebar dynamically
with history_container:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stButton button {
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 15px !important;
    }
    section[data-testid="stSidebar"] .stButton button > div {
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton button p {
        text-align: left !important;
        margin: 0 !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Search History")
    with col2:
        if st.session_state.get("chat_sessions"):
            with st.popover("⋮"):
                if st.button("🗑️ Delete History", use_container_width=True):
                    st.session_state.chat_sessions = []
                    from src.session_store import save_sessions
                    save_sessions([])
                    if st.session_state.get("current_session_id"):
                        st.session_state.qa_history = []
                        st.session_state.current_session_id = None
                    st.rerun()
                    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.qa_history = []
        st.session_state.current_session_id = None
        st.rerun()
        
    st.divider()
        
    if not st.session_state.chat_sessions:
        st.write("No history yet.")
    else:
        st.markdown("""
        <style>
        /* Make the horizontal block relative so we can absolute position inside it */
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker) {
            position: relative;
            gap: 0 !important;
            align-items: center !important;
        }

        /* Remove default Streamlit bottom margins to fix vertical centering */
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker) div[data-testid="element-container"] {
            margin-bottom: 0 !important;
        }
        
        /* Force the first column (history button) to take full width */
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker) > div:nth-child(1) {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Absolutely position the second column (delete button) over the right edge */
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker) > div:nth-child(2) {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            margin-top: -2px; /* Nudge slightly up */
            width: auto !important;
            min-width: 0 !important;
            flex: none !important;
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
            z-index: 10;
        }
        
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker):hover > div:nth-child(2) {
            opacity: 1;
        }

        /* Style the delete button to look like a clean icon instead of a blocky button */
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker) > div:nth-child(2) button {
            background-color: transparent !important;
            border: none !important;
            color: #d9534f !important;
            padding: 2px !important;
            min-height: 0 !important;
            height: auto !important;
            width: 28px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
        }
        
        div[data-testid="stHorizontalBlock"]:has(.history-delete-marker) > div:nth-child(2) button:hover {
            background-color: rgba(217, 83, 79, 0.1) !important;
            border-radius: 4px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for chat in reversed(st.session_state.chat_sessions):
            title = chat["title"][:25] + "..." if len(chat["title"]) > 25 else chat["title"]
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"{title}", key=f"hist_{chat['id']}", use_container_width=True):
                    st.session_state.current_session_id = chat["id"]
                    st.session_state.qa_history = chat["qa_history"]
                    st.rerun()
            with col2:
                st.markdown('<div class="history-delete-marker" style="display:none;"></div>', unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{chat['id']}", use_container_width=True, help="Delete this chat"):
                    # Delete logic for specific history
                    st.session_state.chat_sessions = [c for c in st.session_state.chat_sessions if c['id'] != chat['id']]
                    from src.session_store import save_sessions
                    save_sessions(st.session_state.chat_sessions)
                    
                    # If we deleted the currently active chat, clear the main view
                    if st.session_state.get("current_session_id") == chat["id"]:
                        st.session_state.qa_history = []
                        st.session_state.current_session_id = None
                    st.rerun()
