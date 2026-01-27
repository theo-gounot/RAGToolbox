try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    torch = None
    AutoModel = None
    AutoTokenizer = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

import requests
import numpy as np
from typing import List, Dict, Any, Tuple
from src.land_rag.base import AbstractBaseModule
from src.land_rag.config import RAGConfig, OLLAMA_BASE_URL, LLM_MODEL

class LateChunking(AbstractBaseModule):
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        if AutoTokenizer is None:
            raise ImportError("transformers/torch not installed.")
        self.model_name = "jinaai/jina-embeddings-v2-small-en"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to("cpu")
        self.model.eval()

    def _get_chunk_spans(self, tokens: Dict[str, Any]) -> List[Tuple[int, int]]:
        total_tokens = tokens['input_ids'].size(1)
        chunk_size = self.config.chunking.params.chunk_size
        overlap = self.config.chunking.params.overlap
        
        spans = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            spans.append((start, end))
            if end == total_tokens:
                break
            start += (chunk_size - overlap)
        return spans

    def run(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=8192
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs) 
            token_embeddings = outputs.last_hidden_state 
            
        spans = self._get_chunk_spans(inputs)
        
        chunks_data = []
        input_ids = inputs['input_ids'][0]

        for start, end in spans:
            chunk_token_ids = input_ids[start:end]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            
            span_embeddings = token_embeddings[:, start:end, :]
            
            if span_embeddings.size(1) > 0:
                chunk_embedding = span_embeddings.mean(dim=1).squeeze().tolist()
            else:
                chunk_embedding = [0.0] * 768

            chunks_data.append({
                "content": chunk_text,
                "embedding": chunk_embedding,
                "metadata": {
                    "start_token": start,
                    "end_token": end
                }
            })
            
        return chunks_data

class RecursiveChunking(AbstractBaseModule):
    def run(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        if RecursiveCharacterTextSplitter is None:
            raise ImportError("langchain_text_splitters not installed.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.params.chunk_size,
            chunk_overlap=self.config.chunking.params.overlap
        )
        docs = splitter.create_documents([text])
        
        if TextEmbedding is None:
            raise ImportError("fastembed not installed.")
        embedding_model = TextEmbedding()
        embeddings = list(embedding_model.embed([d.page_content for d in docs]))
        
        return [
            {
                "content": d.page_content,
                "embedding": list(emb),
                "metadata": {}
            }
            for d, emb in zip(docs, embeddings)
        ]

class SemanticChunking(AbstractBaseModule):
    def run(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        if TextEmbedding is None:
            raise ImportError("fastembed not installed.")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        embedding_model = TextEmbedding()
        embeddings = list(embedding_model.embed(sentences)) 
        
        chunks = []
        current_chunk = [sentences[0]]
        current_emb = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]))
            
            if sim < 0.7:
                combined_text = ". ".join(current_chunk) + "."
                avg_emb = np.mean(current_emb, axis=0).tolist()
                chunks.append({"content": combined_text, "embedding": avg_emb, "metadata": {}})
                
                current_chunk = [sentences[i]]
                current_emb = [embeddings[i]]
            else:
                current_chunk.append(sentences[i])
                current_emb.append(embeddings[i])
        
        if current_chunk:
            combined_text = ". ".join(current_chunk) + "."
            avg_emb = np.mean(current_emb, axis=0).tolist()
            chunks.append({"content": combined_text, "embedding": avg_emb, "metadata": {}})
            
        return chunks

class PropositionalChunking(AbstractBaseModule):
    def run(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        chunks = []
        if TextEmbedding is None:
            raise ImportError("fastembed not installed.")
        embedding_model = TextEmbedding()

        for p in paragraphs:
            prompt = f"Extract distinct, independent facts from the following text. Return them as a bulleted list:\n\n{p}"
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
            )
            
            if response.status_code == 200:
                facts = response.json().get("response", "").split('\n')
                facts = [f.strip('- ').strip() for f in facts if f.strip()]
                
                if facts:
                    fact_embeddings = list(embedding_model.embed(facts))
                    for f, emb in zip(facts, fact_embeddings):
                        chunks.append({"content": f, "embedding": list(emb), "metadata": {"origin_paragraph": p[:50]}})
            
        return chunks

class ContextualChunking(AbstractBaseModule):
    def run(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        prompt = f"Summarize the following document in one concise sentence to provide context for its chunks:\n\n{text[:4000]}"
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
        )
        context = response.json().get("response", "").strip() if response.status_code == 200 else ""
        
        if RecursiveCharacterTextSplitter is None:
            raise ImportError("langchain_text_splitters not installed.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.params.chunk_size,
            chunk_overlap=self.config.chunking.params.overlap
        )
        docs = splitter.create_documents([text])
        
        chunks = []
        if TextEmbedding is None:
            raise ImportError("fastembed not installed.")
        embedding_model = TextEmbedding()
        
        texts_to_embed = []
        final_contents = []
        
        for d in docs:
            contextualized_text = f"Context: {context}\nChunk: {d.page_content}"
            texts_to_embed.append(contextualized_text)
            final_contents.append(contextualized_text)
            
        embeddings = list(embedding_model.embed(texts_to_embed))
        
        return [
            {
                "content": c,
                "embedding": list(emb),
                "metadata": {"global_context": context}
            }
            for c, emb in zip(final_contents, embeddings)
        ]

class IndexingFactory:
    @staticmethod
    def get_chunker(config: RAGConfig) -> AbstractBaseModule:
        method = config.chunking.method
        if method == "recursive":
            return RecursiveChunking(config)
        elif method == "semantic":
            return SemanticChunking(config)
        elif method == "late":
            return LateChunking(config)
        elif method == "propositional":
            return PropositionalChunking(config)
        elif method == "contextual":
            return ContextualChunking(config)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
