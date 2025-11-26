import pdfplumber
from llama_index.core import Document

def pdf_to_markdown_tables(pdf_path):
    """
    Extracts text and tables from a PDF using pdfplumber.
    Converts tables into Markdown format to preserve row/column structure
    for the LLM.
    """
    documents = []
    text_content = ""

    print(f"📄 Processing: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 1. Extract Tables
            tables = page.extract_tables()
            
            page_text = ""
            
            if tables:
                for table in tables:
                    # Convert list-of-lists to Markdown Table string
                    # Filter out None values just in case
                    clean_table = [[str(cell) if cell else "" for cell in row] for row in table]
                    
                    # Create Header
                    if clean_table:
                        header = "| " + " | ".join(clean_table[0]) + " |"
                        separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
                        body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_table[1:]])
                        
                        markdown_table = f"\n{header}\n{separator}\n{body}\n"
                        page_text += markdown_table
            else:
                # Fallback: Extract raw text if no tables found (keeps layout)
                page_text += page.extract_text(layout=True) or ""

            # Accumulate content
            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

    # Create a single LlamaIndex Document for the whole file
    # (or you could return one Document per page)
    return [Document(text=text_content, metadata={"filename": pdf_path})]

def load_and_parse_pdfs(data_path="./data"):
    """
    Iterates through the data folder and parses all PDFs.
    """
    import os
    
    all_docs = []
    
    # Check if directory exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created {data_path} folder. Please add PDFs there.")
        return []

    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(data_path, filename)
            docs = pdf_to_markdown_tables(full_path)
            all_docs.extend(docs)
            
    print(f"✅ Successfully parsed {len(all_docs)} documents locally.")
    return all_docs