from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def initialize_index(documents):
    """
    Builds the Vector Store Index from the parsed documents.
    """
    # 1. Setup the LLM (GPT-4o is recommended for reasoning capabilities)
    Settings.llm = OpenAI(model="gpt-4o", temperature=0)
    
    # 2. Setup Embedding Model (Small is cost-effective and sufficient for text)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    print("⚙️  Building Vector Index/Embeddings...")
    
    # 3. Create Index (In-memory for this MVP, switch to ChromaDB for prod)
    index = VectorStoreIndex.from_documents(documents)
    
    return index

def get_query_engine(index):
    """
    Returns a query engine that acts as the interface to ask questions.
    """
    return index.as_query_engine(similarity_top_k=5)