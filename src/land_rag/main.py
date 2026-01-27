import logging
import os
import requests
from typing import Dict, Any, List

try:
    import chromadb
except ImportError:
    chromadb = None

from src.land_rag.config import RAGConfig, RAGOutput, Result, SourceDocument, Metadata, OLLAMA_BASE_URL, LLM_MODEL, EvaluationResult
from src.land_rag.ocr import OCRFactory
from src.land_rag.indexing import IndexingFactory
from src.land_rag.retrieval import RetrievalModule
from src.land_rag.postprocessing import RerankerModule, EvaluationModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Master Implementation of the RAG Pipeline.
    Orchestrates OCR, Indexing, Retrieval, and Generation.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.ocr_module = OCRFactory.get_ocr(config)
        
        # Initialize Database Client (Remote Chroma)
        self.db_client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Initialize Modules
        self.retrieval_module = RetrievalModule(config, self.db_client)
        self.reranker_module = RerankerModule(config)
        self.evaluation_module = EvaluationModule(config)
        
    def ingest_document(self, file_path: str):
        """Processes a file and stores it in Chroma."""
        logger.info(f"Starting ingestion for: {file_path} using {self.config.ocr_method} OCR")
        
        # 1. OCR
        raw_text = self.ocr_module.run(file_path)
        
        # 2. Chunking
        chunker = IndexingFactory.get_chunker(self.config)
        chunks = chunker.run(raw_text)
        
        # 3. Storage
        collection = self.db_client.get_or_create_collection(name="land_rag_collection")
        
        ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c.get("metadata", {}).copy()
            meta["source"] = file_path
            meta["page"] = 0 # Placeholder
            metadatas.append(meta)
        
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Executes the RAG pipeline for a given query.
        """
        logger.info(f"Running pipeline for query: {query}")
        
        # 1. Retrieval
        retrieved_docs = self.retrieval_module.run(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents before reranking.")
        
        # 2. Reranking
        if self.config.retrieval.reranker:
            retrieved_docs = self.reranker_module.run(query, retrieved_docs)
            logger.info("Reranking complete.")
            
        # Top 5 for Generation context
        top_docs = retrieved_docs[:5]
        context_text = "\n\n".join([d["content"] for d in top_docs])
        
        # 3. Generation (via Ollama)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        answer = response.json().get("response", "Error generating answer.") if response.status_code == 200 else "Error"

        # 4. Evaluation
        eval_result_obj = None
        if self.config.eval.enabled:
            logger.info("Running evaluation...")
            scores = self.evaluation_module.run(
                query, 
                answer, 
                [d["content"] for d in top_docs]
            )
            # Flatten scores for schema
            # Ragas returns a dict like {'faithfulness': 0.8, ...}
            eval_result_obj = EvaluationResult(
                scores=scores,
                judge_model=LLM_MODEL
            )

        # 5. Structure Output
        source_docs_objs = [
            SourceDocument(
                content=d["content"],
                metadata=Metadata(
                    source=d.get("metadata", {}).get("source", "unknown"),
                    page=d.get("metadata", {}).get("page", 0),
                    score=d.get("score", 0.0),
                    method_origin=d.get("method_origin", "vector") # Default to vector if unknown to satisfy schema
                )
            )
            for d in top_docs
        ]

        output = RAGOutput(
            query=query,
            result=Result(
                answer=answer,
                source_documents=source_docs_objs,
                evaluation=eval_result_obj
            )
        )
        
        return output.model_dump()
