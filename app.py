import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from src.ingestion import load_and_process_files
from src.rag_engine import initialize_index, get_query_engine

load_dotenv()

# --- UI Configuration ---
st.set_page_config(page_title="Ledger Lens AI", layout="wide", page_icon="💸")

# --- 🎨 PRO UI STYLING (Safe Fix) ---
st.markdown("""
<style>
    /* 1. Global Dark Theme */
    .stApp {
        background-color: #0E1117;
    }
    
    /* 2. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #111827; /* Dark Navy */
        border-right: 1px solid #374151;
    }
    
    /* Force all text in sidebar to be white */
    section[data-testid="stSidebar"] * {
        color: #F3F4F6 !important;
    }

    /* 3. HEADER STYLING (The Fix) */
    /* Instead of hiding it, we make it match the dark background */
    header[data-testid="stHeader"] {
        background-color: #0E1117; /* Matches main app background */
    }
    
    /* Hide the top colored decoration line */
    div[data-testid="stDecoration"] {
        visibility: hidden;
    }
    
    /* 4. BUTTON STYLING */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #1E40AF 0%, #1E3A8A 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }

    /* 5. CHAT BUBBLES */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1F2937;
        border: 1px solid #374151;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #0F172A;
        border: 1px solid #1E3A8A;
    }

    /* 6. Input Field Styling */
    .stTextInput > div > div > input {
        background-color: #1F2937;
        color: white;
        border: 1px solid #374151;
        border-radius: 10px;
    }

    /* Hide Footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- App Logic Starts Here ---

st.title("💸 Ledger Lens AI")
st.caption("Enterprise-Grade Financial Document Analysis | Powered by RAG & GPT-4o")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=60) # Optional Logo placeholder
    st.title("Data Control")
    st.markdown("---")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Statements", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or multiple PDF statements."
    )
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    if uploaded_files:
        if st.button("⚡ Process Documents"):
            with st.spinner("Processing encryption & indexing (Parallel Mode)..."):
                # 1. Create a stable temp directory
                temp_dir = "temp_data"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # 2. Save files
                file_paths = []
                for uploaded_file in uploaded_files:
                    path = os.path.join(temp_dir, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(path)
                
                # 3. Ingest (From src/ingestion.py)
                # This reads the PDFs and makes Markdown tables
                docs = load_and_process_files(file_paths)
                
                # 4. Index (From src/rag_engine.py)
                # This runs the new Parallel Pipeline -> ChromaDB
                index = initialize_index(docs)
                
                # 5. Create Engine
                st.session_state['query_engine'] = get_query_engine(index)
                
                st.toast(f"✅ Indexed {len(docs)} documents into ChromaDB!", icon="🎉")
    
    st.markdown("---")
    
    # Styled 'Tip' Box
    st.markdown("""
    <div style="background-color: #1F2937; padding: 15px; border-radius: 8px; border-left: 4px solid #3B82F6;">
        <span style="color: #9CA3AF; font-size: 0.9em;">💡 <b>Pro Tip:</b> Try asking "List all subscription payments" or "Calculate total spend on Food".</span>
    </div>
    """, unsafe_allow_html=True)

# --- Main Interface ---

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload your statements in the sidebar to get started."}]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your finances..."):
    if 'query_engine' not in st.session_state:
        st.error("Please process documents in the sidebar first!")
    else:
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response (Streaming)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Run Query
            streaming_response = st.session_state['query_engine'].query(prompt)
            
            # Stream the result
            for token in streaming_response.response_gen:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Sources Expander
            with st.expander("🔍 View Source Evidence"):
                sources = streaming_response.source_nodes
                if sources:
                    tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
                    for i, node in enumerate(sources):
                        with tabs[i]:
                            st.caption(f"**Relevance:** {node.score:.3f} | **File:** {node.metadata.get('filename')}")
                            st.markdown(node.node.get_content())