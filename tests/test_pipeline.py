import pytest
from unittest.mock import MagicMock, patch
from src.land_rag.main import RAGPipeline

@patch("src.land_rag.main.OCRFactory")
@patch("src.land_rag.main.IndexingFactory")
@patch("src.land_rag.main.RetrievalModule")
@patch("src.land_rag.main.RerankerModule")
@patch("src.land_rag.main.EvaluationModule")
@patch("src.land_rag.main.requests.post")
def test_pipeline_run(mock_gen, mock_eval, mock_rerank, mock_retrieval, mock_idx, mock_ocr, basic_config):
    # Mock chromadb in main to allow HttpClient mock
    with patch("src.land_rag.main.chromadb") as mock_chroma_main:
        mock_chroma_main.HttpClient.return_value = MagicMock()
        
        pipeline = RAGPipeline(basic_config)
    
        # Mock Ingest
        mock_ocr.get_ocr.return_value.run.return_value = "raw text"
        mock_idx.get_chunker.return_value.run.return_value = [{"content": "c1", "embedding": [0.1], "metadata": {}}]
    
        pipeline.ingest_document("test.pdf")
    
        # Mock Run
        mock_retrieval_inst = mock_retrieval.return_value
        mock_retrieval_inst.run.return_value = [{"id": "doc1", "content": "c1", "score": 0.9}]
    
        mock_gen.return_value.status_code = 200
        mock_gen.return_value.json.return_value = {"response": "The Answer"}
    
        result = pipeline.run_pipeline("Query")
    
        assert result['query'] == "Query"
        assert result['result']['answer'] == "The Answer"