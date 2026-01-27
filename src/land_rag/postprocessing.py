import logging
from typing import List, Dict, Any

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    RerankRequest = None

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
except ImportError:
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

from src.land_rag.base import AbstractBaseModule
from src.land_rag.config import RAGConfig, OLLAMA_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

class RerankerModule(AbstractBaseModule):
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        if Ranker is None:
            self.ranker = None
        else:
            self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./.flashrank_cache")

    def run(self, query: str, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        if self.ranker is None:
            return documents
        if not documents:
            return []
        passages = [
            {"id": doc.get("id", str(i)), "text": doc.get("content"), "meta": doc.get("metadata")}
            for i, doc in enumerate(documents)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        reranked_docs = []
        for res in results:
            reranked_docs.append({
                "content": res["text"],
                "metadata": res["meta"],
                "score": res["score"],
                "method_origin": "reranker"
            })
        return reranked_docs

class EvaluationModule(AbstractBaseModule):
    def run(self, query: str, answer: str, contexts: List[str], **kwargs) -> Dict[str, Any]:
        if evaluate is None or Dataset is None:
            logger.warning("Ragas/Datasets not installed. Skipping evaluation.")
            return {}
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        try:
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
            langchain_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
            langchain_embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=langchain_llm,
                embeddings=langchain_embeddings
            )
            return results
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return {}