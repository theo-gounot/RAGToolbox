import pytest
from unittest.mock import MagicMock, patch
from src.land_rag.retrieval import RetrievalModule
from src.land_rag.config import RAGConfig

@patch("src.land_rag.retrieval.requests.post")
def test_retrieval_flow(mock_post, basic_config):
    # Setup mock for chromadb inside retrieval module
    with patch("src.land_rag.retrieval.chromadb") as mock_chroma:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        module = RetrievalModule(basic_config, mock_client)
        assert module.collection is not None
    
        # Mock Ollama response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "q1\nq2\nq3"}
    
        # Mock Chroma Query results
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['content1']],
            'metadatas': [[{'source': 's1'}]],
            'distances': [[0.1]]
        }
        
        results = module.run("test query")
        assert len(results) > 0
        assert results[0]["content"] == "content1"