import pytest
import os
import json
from unittest.mock import MagicMock, patch
from src.land_rag.config import RAGConfig, ChunkingConfig, ChunkingParams, RetrievalConfig, EvalConfig, KnowledgeGraphConfig
from src.land_rag.indexing import HierarchicalChunking, GraphExtractor
from src.land_rag.retrieval import RetrievalModule
from src.land_rag.main import RAGPipeline
from src.land_rag.dataset_gen import SyntheticTestSetGenerator

@pytest.fixture
def agentic_config():
    return RAGConfig(
        family_name="test_family",
        ocr_method="low",
        chunking=ChunkingConfig(
            method="hierarchical",
            params=ChunkingParams(chunk_size=1500, overlap=200)
        ),
        retrieval=RetrievalConfig(
            methods=["standard", "graph_hop"],
            hybrid_search=False,
            reranker=False
        ),
        eval=EvalConfig(
            enabled=False,
            metrics=[]
        ),
        knowledge_graph=KnowledgeGraphConfig(enabled=True),
        agentic_rag=True
    )

def test_hierarchical_chunking_logic(agentic_config):
    with patch('requests.post') as mock_post:
        # Mock summary response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "This is a test summary."}
        
        chunker = HierarchicalChunking(agentic_config)
        text = "This is a very long text " * 100 # Large enough for parent/child split
        chunks = chunker.run(text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "parent_content" in chunk["metadata"]
            assert "document_summary" in chunk["metadata"]
            assert chunk["metadata"]["document_summary"] == "This is a test summary."

def test_graph_extractor(agentic_config):
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "Apple | is a | company\nSteve Jobs | founded | Apple"}
        
        extractor = GraphExtractor(agentic_config)
        triplets = extractor.run("Apple was founded by Steve Jobs.")
        
        assert len(triplets) == 2
        assert triplets[0]["subject"] == "Apple"
        assert triplets[1]["subject"] == "Steve Jobs"

def test_retrieval_family_naming(agentic_config):
    mock_db = MagicMock()
    # Mocking get_or_create_collection and get_collection to avoid chromadb errors
    mock_db.get_or_create_collection.return_value = MagicMock()
    mock_db.get_collection.return_value = MagicMock()
    
    retrieval = RetrievalModule(agentic_config, mock_db)
    
    assert retrieval.vector_col_name == "test_family_vector"
    assert retrieval.graph_col_name == "test_family_graph"
    mock_db.get_or_create_collection.assert_any_call("test_family_vector")

@patch('src.land_rag.main.requests.post')
@patch('src.land_rag.retrieval.RetrievalModule.run')
def test_agentic_rag_loop(mock_retrieval, mock_post, agentic_config):
    # Mock Decomposition, Critique, and Generation
    mock_post.side_effect = [
        MagicMock(status_code=200, json=lambda: {"response": "sub-query 1\nsub-query 2"}), # Decompose
        MagicMock(status_code=200, json=lambda: {"response": "YES, context is good"}),    # Critique
        MagicMock(status_code=200, json=lambda: {"response": "The final answer."})        # Generation
    ]
    
    mock_retrieval.return_value = [
        {"id": "1", "content": "Context 1", "metadata": {}, "score": 0.9}
    ]
    
    # We need to mock chromadb.HttpClient in main.py or the RAGPipeline initialization
    with patch('chromadb.HttpClient') as mock_http:
        mock_http.return_value = MagicMock()
        pipeline = RAGPipeline(agentic_config)
        
        result = pipeline.run_pipeline("Tell me about Apple.")
        
        assert result["result"]["answer"] == "The final answer."
        assert mock_retrieval.call_count == 2 # Once for each sub-query

def test_synthetic_gen_file_output(agentic_config):
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.get_collection.return_value = mock_col
    mock_col.get.return_value = {
        'documents': ["Some chunk content"],
        'ids': ["id1"],
        'metadatas': [{}]
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "Question: What is it?\nAnswer: It is content."}
        
        gen = SyntheticTestSetGenerator(agentic_config, mock_db)
        output_file = "test_synthetic.jsonl"
        gen.generate_test_set("test_family", num_samples=1, output_file=output_file)
        
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            data = json.loads(f.readline())
            assert data["question"] == "What is it?"
            assert data["ground_truth"] == "It is content."
        
        os.remove(output_file)
