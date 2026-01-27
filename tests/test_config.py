from src.land_rag.config import RAGConfig, RAGOutput
import pytest
from pydantic import ValidationError

def test_valid_rag_config(basic_config):
    # Should not raise
    assert basic_config.ocr_method == "low"

def test_invalid_ocr_method():
    with pytest.raises(ValidationError):
        RAGConfig(
            ocr_method="super_high", # Invalid
            chunking={"method": "recursive", "params": {"chunk_size": 100, "overlap": 10}},
            retrieval={"methods": ["standard"], "hybrid_search": False, "reranker": False},
            eval={"enabled": False, "metrics": []}
        )

def test_rag_output_schema():
    data = {
        "query": "test",
        "result": {
            "answer": "answer",
            "source_documents": [
                {
                    "content": "content",
                    "metadata": {"source": "s", "page": 1, "score": 0.5, "method_origin": "vector"}
                }
            ],
            "evaluation": None
        }
    }
    output = RAGOutput(**data)
    assert output.query == "test"
