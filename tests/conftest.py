import pytest
from unittest.mock import MagicMock
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig

@pytest.fixture
def basic_config():
    return RAGConfig(
        ocr_method="low",
        chunking=ChunkingConfig(
            method="recursive",
            params=ChunkingParams(chunk_size=500, overlap=50)
        ),
        retrieval=RetrievalConfig(
            methods=["standard"],
            hybrid_search=False,
            reranker=False
        ),
        eval=EvalConfig(
            enabled=False,
            metrics=[]
        )
    )

@pytest.fixture
def mock_chroma_client():
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client
