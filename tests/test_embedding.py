from unittest.mock import patch

import pytest

from edgar import SECFiling
from rag.vectorize import (
    TextChunksWithEmbedding,
    _storage_prefix,
    chunk_filing,
)
from rag.vectorize.embedding import GEMINI_EMBEDDING_MODEL
from tests.utils import mock_embedding, mock_file_content


def test_one_filing_chunk_save_load():
    """
    full lifecycle of a filing
    download, chunk, get embedding, save and load
    """
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
        text_chunks = chunk_filing(filing)
        assert text_chunks

    with patch(
        "rag.vectorize.batch_embedding",
        return_value=mock_embedding("1002427/0001133228-24-004879.json"),
    ):
        metadata = {
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
            "model": GEMINI_EMBEDDING_MODEL,
            "dimension": 768,
        }
        chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
        chunks.get_embeddings()
        assert chunks.is_ready() and chunks.save() is None

    restored_chunks = TextChunksWithEmbedding.load(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=GEMINI_EMBEDDING_MODEL,
        dimension=768,
    )
    assert restored_chunks and restored_chunks.is_ready()
    assert chunks.texts == restored_chunks.texts
    assert chunks.embeddings == restored_chunks.embeddings


def test_load_non_existent_chunks():
    with pytest.raises(ValueError):
        TextChunksWithEmbedding.load(
            cik="blah",
            accession_number="blah",
            model=GEMINI_EMBEDDING_MODEL,
            dimension=768,
        )


def test_storage_prefix():
    assert _storage_prefix("gs://bucket/prefix") == ("bucket", "prefix")
    assert _storage_prefix("gs://bucket") == ("bucket", "")
    bucket_name, prefix = _storage_prefix("tmp")
    assert bucket_name is None
    assert prefix.endswith("tmp")
