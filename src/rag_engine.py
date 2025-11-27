import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, PromptTemplate
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import streamlit as st

# --- CONFIGURATION ---
# 1. Setup Persistent DB (Creates a folder 'chroma_db' locally)
db_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db_client.get_or_create_collection("financial_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 2. Define The "Auditor" Prompt
FINANCE_QA_TEMPLATE = PromptTemplate(
    "You are a strict Financial Auditor AI.\n"
    "Context from bank statements:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the question: {query_str}\n\n"
    "RULES:\n"
    "1. Cite specific Dates and Amounts from the context.\n"
    "2. If the answer is missing, say 'Information not found'.\n"
    "3. Show your math for totals.\n"
    "Answer:"
)

def initialize_index(documents):
    """
    Takes documents from ingestion.py and runs them through a Parallel Pipeline
    into ChromaDB.
    """
    # Define the processing steps
    pipeline = IngestionPipeline(
        transformations=[
            # Chunk size 1024 keeps large tables together
            SentenceSplitter(chunk_size=1024, chunk_overlap=100), 
            OpenAIEmbedding(model="text-embedding-3-small")
        ],
        vector_store=vector_store, 
    )

    print(f"🚀 Starting Ingestion for {len(documents)} documents...")
    
    # RUN PIPELINE (This saves vectors to disk automatically)
    # num_workers=4 uses 4 CPU cores for speed (Parallel Processing)
    pipeline.run(documents=documents, num_workers=4)
    
    print("✅ Ingestion Complete. Loading Index from DB...")

    # Connect to the Index sitting in ChromaDB
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    return index

def get_query_engine(index):
    """
    Returns the engine with Reranking enabled.
    """
    # Global Settings for the Query Engine
    Settings.llm = OpenAI(model="gpt-4o", temperature=0, streaming=True)
    
    # Reranker (Anti-Hallucination)
    reranker = FlagEmbeddingReranker(
        top_n=5, 
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        use_fp16=False
    )

    return index.as_query_engine(
        similarity_top_k=20,          # 1. Fetch 20 broad matches
        node_postprocessors=[reranker], # 2. Filter to top 5 best
        text_qa_template=FINANCE_QA_TEMPLATE, # 3. Use Auditor Persona
        streaming=True
    )