"""
Use Case: Granular Fact Extraction for Knowledge Bases.
Best for: Legal documents, policy manuals, and compliance checking.
Features: Propositional Chunking (LLM-based decomposition).
"""
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

config = RAGConfig(
    ocr_method="mid", # Uses DocTR for structured layout detection
    chunking=ChunkingConfig(
        method="propositional", # Every paragraph is broken into independent 'fact' strings by the LLM
        params=ChunkingParams(chunk_size=0, overlap=0)
    ),
    retrieval=RetrievalConfig(
        methods=["standard"],
        hybrid_search=True,
        reranker=False
    ),
    eval=EvalConfig(enabled=True, metrics=["faithfulness", "relevancy"])
)

# Propositional chunking ensures that retrieval returns specific facts
# rather than large, noisy paragraphs.

print("Knowledge Graph Prep: Configured for fact-level propositional chunking.")
