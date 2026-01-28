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
        self.db_client = chromadb.HttpClient(host="10.246.47.192", port=8000)
        
        # Initialize Modules
        # We pass the client, but retrieval module will handle collection selection based on config
        self.retrieval_module = RetrievalModule(config, self.db_client)
        self.reranker_module = RerankerModule(config)
        self.evaluation_module = EvaluationModule(config)
        
    def ingest_document(self, file_path: str, family_name: str = "default", build_graph: bool = False):
        """Processes a file and stores it in Chroma (Vector + Graph)."""
        logger.info(f"Starting ingestion for: {file_path} into family '{family_name}'")
        
        # Update config family name for this operation (contextual)
        self.config.family_name = family_name
        
        # 1. OCR
        raw_text = self.ocr_module.run(file_path)
        
        # 2. Chunking
        chunker = IndexingFactory.get_chunker(self.config)
        chunks = chunker.run(raw_text, page_number=1) # Placeholder page number
        
        # 3. Storage (Vector)
        vector_col_name = f"{family_name}_vector"
        collection = self.db_client.get_or_create_collection(name=vector_col_name)
        
        ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c.get("metadata", {}).copy()
            meta["source"] = file_path
            metadatas.append(meta)
        
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        logger.info(f"Stored {len(chunks)} chunks in {vector_col_name}.")

        # 4. Storage (Graph)
        if build_graph:
            logger.info("Building Knowledge Graph...")
            graph_extractor = IndexingFactory.get_graph_extractor(self.config)
            triplets = graph_extractor.run(raw_text)
            
            if triplets:
                graph_col_name = f"{family_name}_graph"
                graph_collection = self.db_client.get_or_create_collection(name=graph_col_name)
                
                g_ids = [f"triplet_{os.path.basename(file_path)}_{i}" for i in range(len(triplets))]
                g_docs = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets]
                # Embed triplets for "Hop" retrieval
                # We can use the same embedding model or a lighter one. Using same for simplicity.
                # Assuming IndexingFactory has access to embedding model or we instantiate one here.
                # We'll rely on Chroma's default embedding if we don't provide one, 
                # OR we should really use the same embedding logic.
                # For now, let's assume we use FastEmbed if available, or rely on Chroma default.
                # To be consistent, let's embed if we can.
                
                try:
                    from fastembed import TextEmbedding
                    embed_model = TextEmbedding()
                    g_embeddings = list(embed_model.embed(g_docs))
                    g_embeddings = [list(e) for e in g_embeddings]
                except ImportError:
                    g_embeddings = None # Let Chroma use default or fail if remote
                
                g_metadatas = [t for t in triplets] # Store structure in metadata
                
                graph_collection.add(
                    ids=g_ids,
                    documents=g_docs,
                    embeddings=g_embeddings,
                    metadatas=g_metadatas
                )
                logger.info(f"Stored {len(triplets)} triplets in {graph_col_name}.")

    def _decompose_query(self, query: str) -> List[str]:
        prompt = f"Break down the following complex query into 2-3 simpler search sub-queries. Return one per line:\n\n{query}"
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": self.config.ollama_model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                lines = response.json().get("response", "").split('\n')
                return [l.strip() for l in lines if l.strip()]
        except Exception:
            pass
        return [query]

    def _critique_context(self, query: str, context: str) -> bool:
        prompt = (
            f"Query: {query}\n\n"
            f"Context: {context[:4000]}\n\n"
            "Is this context sufficient and relevant to answer the query? "
            "Reply with 'YES' or 'NO' and a brief reason."
        )
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": self.config.ollama_model, "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip().upper()
                return text.startswith("YES")
        except Exception:
            pass
        return True # Default to True if check fails to avoid blocking

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Executes the RAG pipeline for a given query.
        """
        logger.info(f"Running pipeline for query: {query}")
        
        # Check Agentic Mode
        if self.config.agentic_rag:
            logger.info("Agentic Mode Active.")
            sub_queries = self._decompose_query(query)
            logger.info(f"Decomposed into: {sub_queries}")
        else:
            sub_queries = [query]
            
        # Retrieval Loop
        all_retrieved = []
        seen_ids = set()
        
        for sq in sub_queries:
            docs = self.retrieval_module.run(sq)
            for d in docs:
                if d['id'] not in seen_ids:
                    all_retrieved.append(d)
                    seen_ids.add(d['id'])
        
        # Reranking
        if self.config.retrieval.reranker:
            all_retrieved = self.reranker_module.run(query, all_retrieved)
            
        top_docs = all_retrieved[:5]
        context_text = ""
        for d in top_docs:
            source = d.get('metadata', {}).get('source', 'Unknown')
            page = d.get('metadata', {}).get('page_number', '?')
            # Markdown Delimiters
            context_text += f"### Source: {self.config.family_name} | Page: {page}\n{d['content']}\n---\n"
        
        answer = ""
        eval_result_obj = None
        
        # Agentic Critique
        sufficient = True
        if self.config.agentic_rag:
            sufficient = self._critique_context(query, context_text)
            if not sufficient:
                logger.warning("Context critiqued as insufficient.")
                answer = "I could not find sufficient information in the available context to answer your query securely."
        
        if sufficient:
            # Generation
            prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            answer = response.json().get("response", "Error generating answer.") if response.status_code == 200 else "Error"

            # Evaluation (only if sufficient)
            if self.config.eval.enabled:
                logger.info("Running evaluation...")
                scores = self.evaluation_module.run(
                    query, 
                    answer, 
                    [d["content"] for d in top_docs]
                )
                eval_result_obj = EvaluationResult(
                    scores=scores,
                    judge_model=self.config.ollama_model
                )

        # Structure Output
        source_docs_objs = [
            SourceDocument(
                content=d["content"],
                metadata=Metadata(
                    source=d.get("metadata", {}).get("source", "unknown"),
                    page=d.get("metadata", {}).get("page_number", 0),
                    score=d.get("score", 0.0),
                    method_origin=d.get("method_origin", "vector")
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
