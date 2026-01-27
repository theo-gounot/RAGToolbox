"""
Use Case: Evaluating RAG Performance (Benchmarking).
Best for: Developers trying to find the best configuration for a specific dataset.
Features: Ragas Integration + RAG Triad Metrics.
"""
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

def run_benchmark(ocr_type, chunk_type):
    config = RAGConfig(
        ocr_method=ocr_type,
        chunking=ChunkingConfig(
            method=chunk_type,
            params=ChunkingParams(chunk_size=1000, overlap=200)
        ),
        retrieval=RetrievalConfig(
            methods=["standard"],
            hybrid_search=True,
            reranker=True
        ),
        eval=EvalConfig(
            enabled=True, 
            metrics=["faithfulness", "relevancy", "context_precision"]
        )
    )

    pipeline = RAGPipeline(config)
    # pipeline.ingest_document("test_data.pdf")
    # result = pipeline.run_pipeline("What are the core metrics?")
    
    # print(f"Method: {ocr_type} + {chunk_type}")
    # print(f"Scores: {result['result']['evaluation']['scores']}")

if __name__ == "__main__":
    print("Benchmark Suite: Demonstrating Ragas integration for 'Grid Search' performance testing.")
    # Example grid search:
    # run_benchmark("low", "recursive")
    # run_benchmark("high", "late")
