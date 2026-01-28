"""
Use Case: Simple, Fast RAG for Text-based PDFs.
Best for: Standard reports, books, and text-heavy documents.
"""
import os
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

def main():
    # 1. Configuration: Optimized for speed and low CPU usage
    config = RAGConfig(
        ocr_method="low",  # Fast pypdfium2 extraction
        chunking=ChunkingConfig(
            method="recursive", # Standard paragraph-based splitting
            params=ChunkingParams(chunk_size=800, overlap=100)
        ),
        retrieval=RetrievalConfig(
            methods=["standard"],
            hybrid_search=False,
            reranker=False
        ),
        eval=EvalConfig(enabled=False, metrics=[])
    )

    # 2. Initialize Pipeline
    pipeline = RAGPipeline(config)

    # 3. Process a document
    # Note: Replace with a real path to a PDF
    pdf_path = "/home/localuser/RAGToolbox/examples/2509.00642v1.pdf"
    if os.path.exists(pdf_path):
        pipeline.ingest_document(pdf_path)
        
        # 4. Ask a question
        query = "What is the summary of this document?"
        result = pipeline.run_pipeline(query)
        
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['result']['answer']}")
    else:
        print(f"Please provide a valid PDF at {pdf_path} to run this example.")

if __name__ == "__main__":
    main()

