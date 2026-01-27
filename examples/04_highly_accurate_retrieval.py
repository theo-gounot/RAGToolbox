"""
Use Case: Needle-in-a-haystack or Vague Queries.
Best for: Large knowledge bases where queries might not match document keywords.
Features: HyDE (Hypothetical Answer) + Multi-Query + Hybrid Search.
"""
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

config = RAGConfig(
    ocr_method="low",
    chunking=ChunkingConfig(
        method="semantic", # Splits on topic shifts (cosine distance)
        params=ChunkingParams(chunk_size=0, overlap=0) # Semantic doesn't use fixed size
    ),
    retrieval=RetrievalConfig(
        methods=["multi_query", "hyde"], 
        # HyDE: Generates a 'fake' answer to use as a better embedding vector
        # Multi-query: Tries 3 different ways to ask the question
        hybrid_search=True,
        reranker=True # Re-sorts the combined pile
    ),
    eval=EvalConfig(enabled=True, metrics=["relevancy", "context_precision"])
)

# Use this when users ask vague questions like "Tell me about the stuff related to project X"
# where keyword search might fail.

print("Advanced Retrieval: Configured for maximum coverage using HyDE and Multi-query.")
