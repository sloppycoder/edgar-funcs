from unittest.mock import patch

from edgar import SECFiling
from rag import (
    TRUSTEE_COMP_QUERIES,
    TextChunksWithEmbedding,
    _storage_prefix,
    chunk_filing,
    load_chunks,
    save_chunks,
)
from rag.embedding import GEMINI_EMBEDDING_MODEL
from tests.utils import mock_embedding, mock_file_content


def test_get_queries_embedding():
    with patch(
        "rag.batch_embedding",
        return_value=mock_embedding("trustee_comp_queries.json"),
    ):
        queries_pair = TextChunksWithEmbedding(TRUSTEE_COMP_QUERIES)
        assert not queries_pair.is_ready()  # not ready without loading embeddings

        queries_pair.get_embeddings()
        assert queries_pair.is_ready()


def test_one_filing_full_lifecyele():
    """
    full lifecycle of a filing
    download, chunk, get embedding, save and load
    """
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
        metadata = {
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
            "model": GEMINI_EMBEDDING_MODEL,
            "dimension": 768,
        }
        text_chunks = chunk_filing(filing)
        assert text_chunks

    with patch(
        "rag.batch_embedding",
        return_value=mock_embedding("1002427/0001133228-24-004879.json"),
    ):
        chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
        chunks.get_embeddings()
        assert chunks.is_ready()

    save_chunks(chunks)
    restored_chunks = load_chunks(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=GEMINI_EMBEDDING_MODEL,
        dimension=768,
    )
    assert restored_chunks.is_ready()
    assert chunks.texts == restored_chunks.texts
    assert chunks.embeddings == restored_chunks.embeddings


def test_storage_prefix():
    assert _storage_prefix("gs://bucket/prefix") == ("bucket", "prefix")
    assert _storage_prefix("gs://bucket") == ("bucket", "")
    bucket_name, prefix = _storage_prefix("tmp")
    assert bucket_name is None
    assert prefix.endswith("tmp")
