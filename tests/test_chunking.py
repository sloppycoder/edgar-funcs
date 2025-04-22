import os
from unittest.mock import patch

import pytest

from edgar_funcs.edgar import SECFiling
from edgar_funcs.rag.vectorize import TextChunksWithEmbedding
from edgar_funcs.rag.vectorize.chunking import (
    _is_line_empty,
    chunk_text,
    trim_html_content,
)
from tests.utils import mock_file_content

if "CLOUD_BUILD" in os.environ or "BUILDER_OUTPUT" in os.environ:
    pytest.skip("CLI tests should not run in Cloud Build", allow_module_level=True)


@patch("edgar_funcs.edgar.edgar_file")
def test_chunk_html_filing(mock_edgar_file):
    mock_edgar_file.side_effect = mock_file_content
    filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
    filing_path, filing_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

    assert filing_path.endswith(".html") or filing_path.endswith(".htm")

    trimmed_html = trim_html_content(filing_content)
    chunks = chunk_text(trimmed_html, method="spacy")

    assert len(chunks) == 992
    assert all(chunk and len(chunk) > 10 for chunk in chunks)


@patch("edgar_funcs.edgar.edgar_file")
def test_chunk_txt_filing(mock_edgar_file):
    mock_edgar_file.side_effect = mock_file_content
    filing = SECFiling(
        cik="1201932",
        accession_number="0000950136-04-001365",
        prefer_index_headers=False,
    )
    filing_path, filing_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

    assert filing_path.endswith(".txt")

    chunks = chunk_text(filing_content, method="spacy")

    # uncomment to save chunks pickle file to mockdata folder
    # _save_chunks_mockdata(filing, chunks)

    assert len(chunks) == 379
    assert all(chunk and len(chunk) > 10 for chunk in chunks)


@patch("edgar_funcs.edgar.edgar_file")
def test_chunk_filing_with_difficult_table(mock_edgar_file):
    mock_edgar_file.side_effect = mock_file_content
    filing = SECFiling(cik="1274676", accession_number="0000919574-24-004904")
    filing_path, filing_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

    assert filing_path.endswith(".html") or filing_path.endswith(".htm")

    trimmed_html = trim_html_content(filing_content)
    chunks = chunk_text(trimmed_html, method="spacy")

    assert len(chunks) == 715
    assert all(chunk and len(chunk) > 10 for chunk in chunks)

    chunk_of_table = [chunk for chunk in chunks if "2,691" in chunk][0]
    assert "Name of Trustee |  Aggregate Compensation" in chunk_of_table
    assert "Emilie D. Wrapp" in chunk_of_table


def test_is_line_empty():
    assert _is_line_empty("   ")
    assert _is_line_empty(" -83-")
    assert _is_line_empty(" wo- wb- xp")
    assert not _is_line_empty(" word ")


def _save_chunks_mockdata(filing: SECFiling, text_chunks: list[str]):
    metadata = {
        "cik": filing.cik,
        "accession_number": filing.accession_number,
        "date_filed": filing.date_filed,
    }
    embedding_model, embedding_dimension = "text-embedding-005", 768
    new_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
    new_chunks.get_embeddings(model=embedding_model, dimension=embedding_dimension)
    new_chunks.save()
