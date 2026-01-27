"""
Use Case: Visually Complex or Scanned Documents.
Best for: Invoices, brochures, infographics, and legacy scanned PDFs.
Features: VLM OCR (Vision Model) + Contextual Chunking.
"""
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

config = RAGConfig(
    ocr_method="VLM",  # PDF -> Images -> Remote VLM (Vision-Language Model)
    chunking=ChunkingConfig(
        method="contextual", # Generates a global doc summary and prepends to every chunk
        params=ChunkingParams(chunk_size=500, overlap=50)
    ),
    retrieval=RetrievalConfig(
        methods=["sentence_window"], # Retrieves original chunk + context window
        hybrid_search=True,
        reranker=True
    ),
    eval=EvalConfig(enabled=False, metrics=[])
)

pipeline = RAGPipeline(config)

# This would convert each page to a JPEG and send it to Ollama
# pipeline.ingest_document("scanned_invoice.pdf")
# pipeline.run_pipeline("What is the invoice number and total amount?")

print("VLM Example: Configured to handle vision-based extraction using remote Ollama.")
