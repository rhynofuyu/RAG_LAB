class Config:
    DATABASE_PATH = 'users.db'
    PINECONE_INDEX_NAME = "rag-lab-index"
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "gemini-2.5-flash"
    
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_TOP_K = 4
    DEFAULT_QUERY_VARIATIONS = 3
    DEFAULT_TOP_K_PER_QUERY = 3
    
    PINECONE_DIMENSION = 768
    PINECONE_METRIC = "cosine"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
