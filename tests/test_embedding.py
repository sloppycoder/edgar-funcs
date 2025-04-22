from unittest.mock import patch

import pytest

from edgar_funcs.edgar import SECFiling
from edgar_funcs.rag.vectorize import (
    TextChunksWithEmbedding,
    _storage_prefix,
)
from edgar_funcs.rag.vectorize.chunking import CHUNK_ALORITHM_VERSION
from tests.utils import mock_file_content, mock_json_dict

embedding_model, embedding_dimension = "text-embedding-005", 768


@patch("edgar_funcs.edgar.edgar_file")
@patch("edgar_funcs.rag.vectorize.batch_embedding")
def test_one_filing_chunk_save_load(mock_batch_embedding, mock_edgar_file):
    """
    full lifecycle of a filing
    download, chunk, get embedding, save and load
    """
    mock_edgar_file.side_effect = mock_file_content
    mock_batch_embedding.return_value = mock_json_dict(
        f"embeddings/{embedding_model}_{embedding_dimension}/1002427/0001133228-24-004879.json"
    )

    filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
    # text_chunks = chunk_filing(filing)
    # chunk_filings runs too slow, use mock data instead
    text_chunks = mock_json_dict("chunking/4/1002427/0001133228-24-004879.json")
    assert text_chunks

    chunks = TextChunksWithEmbedding(
        text_chunks,
        metadata={
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
            "chunk_algo_version": CHUNK_ALORITHM_VERSION,
        },
    )
    chunks.get_embeddings(model=embedding_model, dimension=embedding_dimension)
    assert chunks.is_ready() and chunks.save() is None

    restored_chunks = TextChunksWithEmbedding.load(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=embedding_model,
        dimension=embedding_dimension,
        chunk_algo_version=CHUNK_ALORITHM_VERSION,
    )
    assert restored_chunks and restored_chunks.is_ready()
    assert chunks.texts == restored_chunks.texts
    assert chunks.embeddings == restored_chunks.embeddings


def test_load_non_existent_chunks():
    with pytest.raises(ValueError):
        TextChunksWithEmbedding.load(
            cik="blah",
            accession_number="blah",
            model=embedding_model,
            dimension=embedding_dimension,
            chunk_algo_version=CHUNK_ALORITHM_VERSION,
        )


def test_storage_prefix():
    assert _storage_prefix("gs://bucket/prefix") == ("bucket", "prefix")
    assert _storage_prefix("gs://bucket") == ("bucket", "")
    bucket_name, prefix = _storage_prefix("tmp")
    assert bucket_name is None
    assert prefix.endswith("tmp")
