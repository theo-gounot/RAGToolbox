import pytest
from unittest.mock import MagicMock, patch
from src.land_rag.indexing import IndexingFactory, RecursiveChunking, LateChunking
from src.land_rag.config import RAGConfig

def test_indexing_factory(basic_config):
    basic_config.chunking.method = "recursive"
    with patch("src.land_rag.indexing.RecursiveCharacterTextSplitter", MagicMock()):
        chunker = IndexingFactory.get_chunker(basic_config)
        assert isinstance(chunker, RecursiveChunking)

def test_recursive_chunking(basic_config):
    with patch("src.land_rag.indexing.RecursiveCharacterTextSplitter") as MockSplitter, \
         patch("src.land_rag.indexing.TextEmbedding") as MockEmbed:
        
        # Mock Splitter
        mock_splitter_inst = MockSplitter.return_value
        mock_splitter_inst.create_documents.return_value = [MagicMock(page_content="c1")]
        
        # Mock Embed
        mock_model = MockEmbed.return_value
        mock_model.embed.return_value = [[0.1]*384]
        
        chunker = RecursiveChunking(basic_config)
        chunks = chunker.run("Hello world")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "c1"

@patch("src.land_rag.indexing.AutoTokenizer")
@patch("src.land_rag.indexing.AutoModel")
@patch("src.land_rag.indexing.torch")
def test_late_chunking_run(mock_torch, mock_model_cls, mock_tokenizer_cls, basic_config):
    # Ensure LateChunking doesn't fail on init due to None check
    with patch("src.land_rag.indexing.AutoTokenizer", mock_tokenizer_cls):
        chunker = LateChunking(basic_config)
    
    mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(size=lambda x: 10),
        "attention_mask": MagicMock()
    }
    mock_tokenizer.return_value['input_ids'][0] = MagicMock()
    
    mock_model = mock_model_cls.from_pretrained.return_value
    mock_out = MagicMock()
    mock_out.last_hidden_state = MagicMock()
    # Mock slicing
    span_mock = MagicMock()
    span_mock.size.return_value = 10
    mock_out.last_hidden_state.__getitem__.return_value = span_mock
    mock_model.return_value = mock_out

    # Mock torch.no_grad context manager
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()

    chunks = chunker.run("Sample text")
    assert isinstance(chunks, list)