"""
Use Case: Deep Scientific Paper Analysis.
Best for: Academic papers, math-heavy PDFs, and technical documentation.
Features: Nougat OCR + Late Chunking (Contextual Embeddings).
"""
import os
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

def main():
    # 1. Configuration: High precision
    config = RAGConfig(
        ocr_method="high",  # Uses Nougat (Subprocess) to extract formulas and Markdown
        chunking=ChunkingConfig(
            method="late", # Embeds full doc first, then pools; preserves global context
            params=ChunkingParams(chunk_size=1024, overlap=256)
        ),
        retrieval=RetrievalConfig(
            methods=["multi_query"], # Expansion to capture technical terminology variations
            hybrid_search=True,      # Vector + BM25 is critical for technical terms
            reranker=True            # FlashRank re-scoring for precision
        ),
        eval=EvalConfig(enabled=True, metrics=["faithfulness"])
    )

    pipeline = RAGPipeline(config)

    pdf_path = "physics_paper.pdf"
    if os.path.exists(pdf_path):
        pipeline.ingest_document(pdf_path)
        
        # Complex technical query
        query = "Explain the derivation of the Schödinger equation in section 3."
        result = pipeline.run_pipeline(query)
        
        print(f"Answer: {result['result']['answer']}")
        # Check evaluation score
        if result['result']['evaluation']:
            print(f"Faithfulness Score: {result['result']['evaluation']['scores']['faithfulness']}")
    else:
        print("File physics_paper.pdf not found. Skipping execution.")

if __name__ == "__main__":
    main()
