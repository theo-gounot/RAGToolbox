# LAND-UFRJ RAG Toolbox

A modular Python RAG framework designed for CPU-first execution, offloading LLM/VLM tasks to a remote Ollama server.

## Features

- **Modular Architecture**: Independent modules for OCR, Indexing, Retrieval, and Post-processing.
- **CPU Optimized**: Uses `fastembed`, `flashrank`, and `pypdfium2` for efficient local processing.
- **Advanced Strategies**:
    - **OCR**: Support for `Nougat` (Formulas), `DocTR` (Layouts), and `VLM` (Visual).
    - **Indexing**: `Late Chunking`, `Semantic`, `Propositional`, `Contextual`.
    - **Retrieval**: Hybrid Search (BM25 + Vector) with RRF fusion, Multi-Query, HyDE.
    - **Eval**: Integrated Ragas evaluation using Ollama.

## Setup

1. **Install uv**:
   ```bash
   pip install uv
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   # or manually
   pip install -e .
   ```

3. **Configure Environment**:
   Edit `.env`:
   ```env
   OLLAMA_BASE_URL="http://localhost:11434"
   LLM_MODEL="llama3.1:8b"
   VLM_MODEL="llava:7b"
   ```

4. **Run Ollama**:
   Ensure your Ollama server is running and the models are pulled:
   ```bash
   ollama pull llama3.1:8b
   ollama pull llava:7b
   ```

## Usage

```python
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig
from src.land_rag.main import RAGPipeline

# Define Configuration
config = RAGConfig(
    ocr_method="low",
    chunking=ChunkingConfig(
        method="late",
        params=ChunkingParams(chunk_size=1000, overlap=200)
    ),
    retrieval=RetrievalConfig(
        methods=["standard", "multi_query"],
        hybrid_search=True,
        reranker=True
    ),
    eval=EvalConfig(
        enabled=True,
        metrics=["faithfulness", "relevancy"]
    )
)

# Initialize Pipeline
pipeline = RAGPipeline(config)

# Ingest
pipeline.ingest_document("path/to/document.pdf")

# Run
result = pipeline.run_pipeline("What is the main conclusion of the paper?")
print(result)
```
# RAGToolbox
