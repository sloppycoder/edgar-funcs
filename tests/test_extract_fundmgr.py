from unittest.mock import patch

import pytest  # noqa: F401

from edgar_funcs.edgar import SECFiling
from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from edgar_funcs.rag.vectorize import TextChunksWithEmbedding
from edgar_funcs.rag.vectorize.chunking import chunk_text, trim_html_content
from tests.utils import mock_file_content

embedding_model, embedding_dimension, extraction_model = (
    "text-embedding-3-small",
    1536,
    "gemini-2.0-flash",
)


def test_extract_fundmgr_ownership():
    cik, accession_number, chunk_algo_version = "1002427", "0001133228-24-004879", "4"

    mock_response = (
        f"response/{extraction_model}/{cik}/{accession_number}_fundmgr_ownership.txt"
    )
    with patch(
        "edgar_funcs.rag.extract.fundmgr.ask_model",
        return_value=mock_file_content(mock_response),
    ):
        result = extract_fundmgr_ownership_from_filing(
            cik=cik,
            accession_number=accession_number,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            model=extraction_model,
            chunk_algo_version=chunk_algo_version,
        )
        assert result and result["ownership_info"]
        managers = result["ownership_info"]["managers"]
        assert len(managers) == 6
        assert (
            managers[0]["name"] == "Dennis P. Lynch"
            and managers[0]["ownership_range"] == "Over 1,000,000"
        )


@pytest.mark.skip(reason="local test only")
def test_full_extract_fundmgr_ownership():
    cik, accession_number, chunk_algo_version = "1342947", "0001213900-24-018034", "4"

    try:
        TextChunksWithEmbedding.load(
            cik=cik,
            accession_number=accession_number,
            chunk_algo_version=chunk_algo_version,
            model=embedding_model,
            dimension=embedding_dimension,
        )
    except ValueError:
        _chunk_and_get_embeddings(cik, accession_number, chunk_algo_version)

    result = extract_fundmgr_ownership_from_filing(
        cik=cik,
        accession_number=accession_number,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        model=extraction_model,
        chunk_algo_version=chunk_algo_version,
    )
    assert result and result["ownership_info"]


def _chunk_and_get_embeddings(cik: str, accession_number: str, chunk_algo_version: str):
    filing = SECFiling(
        cik=cik,
        accession_number=accession_number,
    )
    filing_path, filing_content = filing.get_doc_content("485BPOS", max_items=1)[0]

    assert filing_path.endswith(".htm")

    if filing_path.endswith((".html", ".htm")):
        text_content = trim_html_content(filing_content)
    else:
        text_content = filing_content

    text_chunks = chunk_text(text_content, method="spacy", chunk_size=1000)
    chunks = TextChunksWithEmbedding(
        text_chunks,
        metadata={
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
            "chunk_algo_version": chunk_algo_version,
        },
    )
    chunks.get_embeddings(model=embedding_model, dimension=embedding_dimension)
    chunks.save()
