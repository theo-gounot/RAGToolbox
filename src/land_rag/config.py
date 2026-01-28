import os
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment Variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
VLM_MODEL = os.getenv("VLM_MODEL", "llava:7b")

# --- Configuration Schema ---

class ChunkingParams(BaseModel):
    chunk_size: int = 1000
    overlap: int = 200

class ChunkingConfig(BaseModel):
    method: Literal["recursive", "semantic", "late", "propositional", "contextual", "hierarchical"] # Added hierarchical
    params: ChunkingParams

class RetrievalConfig(BaseModel):
    methods: List[Literal["standard", "sentence_window", "multi_query", "hyde", "graph_hop"]] # Added graph_hop
    hybrid_search: bool
    reranker: bool

class EvalConfig(BaseModel):
    enabled: bool
    metrics: List[Literal["faithfulness", "relevancy", "context_precision"]]

class KnowledgeGraphConfig(BaseModel):
    enabled: bool = False
    triplet_extraction_model: str = "llama3.1:8b" # Model to use for extracting triplets

class RAGConfig(BaseModel):
    family_name: str = "default" # Family name for collection grouping
    ocr_method: Literal["low", "mid", "high", "VLM"]
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    eval: EvalConfig
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    agentic_rag: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5" # Changed from "fastembed"
    ollama_model: str = LLM_MODEL

# --- Output Schema ---

class Metadata(BaseModel):
    source: str
    page: int
    score: float
    method_origin: Literal["vector", "bm25"]

class SourceDocument(BaseModel):
    content: str
    metadata: Metadata

class EvaluationScores(BaseModel):
    faithfulness: float
    relevancy: float
    context_precision: Optional[float] = None # Added optional based on metric list but example shows only 2

class EvaluationResult(BaseModel):
    scores: Dict[str, float] # Using Dict to be flexible or strictly EvaluationScores
    judge_model: str

class Result(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
    evaluation: Optional[EvaluationResult] = None

class RAGOutput(BaseModel):
    query: str
    result: Result
