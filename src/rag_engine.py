from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.node_parser import SentenceSplitter
import streamlit as st

# 1. DEFINE A "STRICT AUDITOR" PROMPT
# This forces the model to think step-by-step and cite the exact row.
FINANCE_QA_TEMPLATE = PromptTemplate(
    "You are a strict Financial Auditor AI. Your goal is precision.\n"
    "We have provided context information below from a bank statement.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this context, answer the question: {query_str}\n\n"
    "RULES:\n"
    "1. CITATIONS: When listing transactions, you MUST explicitly state the Date, Description, and Amount from the context.\n"
    "2. NO GUESSING: If the answer is not in the context, say 'I cannot find that information in the documents provided'.\n"
    "3. MATH: If asked to sum numbers, list the individual numbers you are adding first, then show the total.\n"
    "4. FORMATTING: Use Markdown tables for lists of transactions.\n"
    "\n"
    "Answer:"
)

@st.cache_resource(show_spinner=False)
def initialize_index(_documents):
    # 2. OPTIMIZE FOR TABLES (Context Window)
    # Bank tables are long. We increase chunk_size to 1024 (from default 512) 
    # to keep rows together.
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    
    # Enable Streaming for speed
    Settings.llm = OpenAI(model="gpt-4o", temperature=0, streaming=True)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    return VectorStoreIndex.from_documents(_documents)

def get_query_engine(index):
    # 3. RERANKER (Precision Layer)
    # This filters out "irrelevant" matches that confuse the LLM.
    reranker = FlagEmbeddingReranker(
        top_n=7,  # Give the LLM 7 highly relevant chunks
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        use_fp16=False
    )

    query_engine = index.as_query_engine(
        similarity_top_k=20,  # Fetch 20 broad matches first
        node_postprocessors=[reranker],
        text_qa_template=FINANCE_QA_TEMPLATE, # Inject our Auditor Persona
        streaming=True
    )
    
    return query_engine