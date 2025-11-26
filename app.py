import streamlit as st
import os
from dotenv import load_dotenv
from src.ingestion import load_and_parse_pdfs
from src.rag_engine import initialize_index, get_query_engine

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Fin-Analyzer Local", layout="wide")

st.title("💸 AI-Powered Bank Statement Analyzer (Local Parser)")
st.markdown("""
This tool uses a **Custom pdfplumber Parser** to extract financial tables locally 
and **GPT-4** (via RAG) to analyze them.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # We only need OpenAI key now
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OpenAI API Key in .env")
        
    st.info("Step 1: Place your Bank Statement PDFs in the `data/` folder.")
    
    if st.button("🔄 Process Documents"):
        with st.spinner("Parsing PDFs locally (Privacy Friendly)..."):
            try:
                # 1. Ingest (Local pdfplumber)
                docs = load_and_parse_pdfs()
                
                if not docs:
                    st.warning("No PDFs found in data/ folder!")
                else:
                    # 2. Index
                    index = initialize_index(docs)
                    # 3. Store in session state
                    st.session_state['query_engine'] = get_query_engine(index)
                    st.success(f"Indexed {len(docs)} documents!")
            except Exception as e:
                st.error(f"Error: {e}")

# Main Chat Interface
if 'query_engine' in st.session_state:
    st.divider()
    query = st.text_input("Ask a question about your finances:", 
                          placeholder="e.g., How much did I spend on Uber in January?")
    
    if query:
        with st.spinner("Analyzing..."):
            response = st.session_state['query_engine'].query(query)
            
            st.markdown("### 🤖 Analysis")
            st.write(response.response)
            
            with st.expander("🔍 View Retrieved Context (How the AI saw your data)"):
                for node in response.source_nodes:
                    st.caption(f"Score: {node.score:.2f}")
                    # This will show the Markdown table structure we created!
                    st.code(node.node.get_content())
else:
    st.info("Please process documents from the sidebar to begin.")