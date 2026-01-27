import json
import logging
import requests
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except ImportError:
    chromadb = None
    Collection = None

from src.land_rag.config import RAGConfig, OLLAMA_BASE_URL, LLM_MODEL
from src.land_rag.utils.rrf import reciprocal_rank_fusion
from src.land_rag.base import AbstractBaseModule

logger = logging.getLogger(__name__)

class RetrievalModule(AbstractBaseModule):
    def __init__(self, config: RAGConfig, db_client: Any):
        super().__init__(config)
        self.db_client = db_client
        if chromadb:
            self.collection = self.db_client.get_or_create_collection("land_rag_collection")
        else:
            self.collection = None
        
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []
        self.doc_ids: List[str] = []

    def _build_bm25(self):
        if not self.collection:
            logger.warning("No collection available for BM25.")
            return
        results = self.collection.get() 
        documents = results['documents']
        ids = results['ids']
        if not documents:
            return
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_ids = ids
        self.bm25_corpus = documents

    def _generate_multi_query(self, query: str) -> List[str]:
        prompt = f"Generate 3 different search query variations for the following user question to improve retrieval coverage. Return only the queries, one per line:\n\n{query}"
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            text = response.json().get("response", "")
            return [q.strip() for q in text.split('\n') if q.strip()]
        return [query]

    def _generate_hyde(self, query: str) -> str:
        prompt = f"Write a hypothetical answer to the following question. This answer will be used to semantic search for relevant documents. Be detailed:\n\nQuestion: {query}\n\nAnswer:"
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("response", query)
        return query

    def run(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        queries = [query]
        if "multi_query" in self.config.retrieval.methods:
            queries = self._generate_multi_query(query)
        search_query = query
        if "hyde" in self.config.retrieval.methods:
            search_query = self._generate_hyde(query)
        all_results = []
        vector_results = []
        for q in queries:
            q_text = search_query if "hyde" in self.config.retrieval.methods else q
            res = self.collection.query(query_texts=[q_text], n_results=10)
            for i in range(len(res['ids'][0])):
                vector_results.append({
                    "id": res['ids'][0][i],
                    "content": res['documents'][0][i],
                    "metadata": res['metadatas'][0][i] if res['metadatas'] else {},
                    "score": 1.0 - res['distances'][0][i] if res['distances'] else 0.0,
                    "method_origin": "vector"
                })
        all_results.append(vector_results)
        if self.config.retrieval.hybrid_search:
            if not self.bm25:
                self._build_bm25()
            if self.bm25:
                tokenized_query = query.split(" ")
                bm25_scores = self.bm25.get_scores(tokenized_query)
                top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
                bm25_results = []
                for idx in top_n:
                    bm25_results.append({
                        "id": self.doc_ids[idx],
                        "content": self.bm25_corpus[idx],
                        "metadata": {},
                        "score": float(bm25_scores[idx]),
                        "method_origin": "bm25"
                    })
                all_results.append(bm25_results)
        merged_results = reciprocal_rank_fusion(all_results)
        if "sentence_window" in self.config.retrieval.methods:
            for doc in merged_results:
                if "window_content" in doc.get("metadata", {}):
                    doc["content"] = doc["metadata"]["window_content"]
        return merged_results[:20]
