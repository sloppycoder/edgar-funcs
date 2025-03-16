from unittest.mock import patch

from edgar import SECFiling
from rag import TextChunkWithEmbedding, chunk_filing
from tests.utils import mock_embedding, mock_file_content


def test_get_queries_embedding():
    trustee_comp_queries = [
        "Trustee Compensation Structure and Amount",
        "Independent Director or Trustee Compensation Table",
        "Board Director or Intereed Person Compensation Details with Amount",
        "Interested Person Compensation Remuneration Detailed Amount",
    ]
    with patch(
        "rag.batch_embedding",
        return_value=mock_embedding("trustee_comp_queries.json"),
    ):
        queries_pair = TextChunkWithEmbedding(trustee_comp_queries)
        assert not queries_pair.is_ready()  # not ready without loading embeddings

        queries_pair.get_embeddings()
        assert queries_pair.is_ready()


def test_get_filing_embedding():
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="883622", accession_number="0001137439-24-001242")
        metadata = {
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
        }
        text_chunks = chunk_filing(filing)
        assert text_chunks

    with patch(
        "rag.batch_embedding",
        return_value=mock_embedding("883622/0001137439-24-001242.json"),
    ):
        chunks = TextChunkWithEmbedding(text_chunks, metadata=metadata)
        chunks.get_embeddings()
        assert chunks.is_ready()
