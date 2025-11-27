import pdfplumber
from llama_index.core import Document
import streamlit as st
import os

# CACHE: This function will NOT re-run if the file_refs (file names) haven't changed
@st.cache_resource(show_spinner=False)
def load_and_process_files(file_refs):
    """
    Parses PDFs only once. file_refs is a list of file paths.
    """
    all_docs = []
    print("⚡ Parsing Documents (Fresh Run)...")
    
    for pdf_path in file_refs:
        text_content = ""
        filename = os.path.basename(pdf_path)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 1. Extract Tables to Markdown (Speed optimization: simplified logic)
                    tables = page.extract_tables()
                    page_text = ""
                    
                    if tables:
                        for table in tables:
                            # Fast list comp to clean None values
                            clean_table = [[str(cell) or "" for cell in row] for row in table]
                            if clean_table:
                                # Quick Markdown construction
                                header = "| " + " | ".join(clean_table[0]) + " |"
                                separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
                                body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_table[1:]])
                                page_text += f"\n{header}\n{separator}\n{body}\n"
                    
                    # 2. Fallback text
                    page_text += f"\n{page.extract_text() or ''}\n"
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            all_docs.append(Document(text=text_content, metadata={"filename": filename}))
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return all_docs