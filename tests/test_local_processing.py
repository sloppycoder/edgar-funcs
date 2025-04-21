# all tests in this file are meant for local debugging only

import os

import pytest  # noqa: F401

from edgar_funcs.edgar import SECFiling
from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from edgar_funcs.rag.vectorize import (
    TextChunksWithEmbedding,
)
from edgar_funcs.rag.vectorize.chunking import chunk_text, trim_html_content
from func_helpers import _get_lock_blob, delete_lock, write_lock

embedding_model, embedding_dimension, extraction_model = (
    "text-embedding-3-small",
    1536,
    "gemini-2.0-flash",
)


@pytest.mark.skip(reason="local testing only")
def test_parse_one_filing():
    # this file has 2 documents of type 485BPOS, one htm another pdf
    # cik, accession_number = "1141819", "0000894189-10-001730"
    cik, accession_number = "1331971", "0000950123-10-116389"
    filing = SECFiling(cik=cik, accession_number=accession_number)
    html_path, html_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

    assert filing.cik == cik


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


@pytest.mark.skip(reason="local testing only")
def test_full_chunking_and_embedding():
    cik, accession_number, chunk_algo_version = "1006415", "0001104659-20-088997", "4"
    chunks = _chunk_and_get_embeddings(cik, accession_number, chunk_algo_version)
    assert len(chunks.texts) > 0


@pytest.mark.skip(reason="local testing only")
def test_lock_file():
    os.environ["STORAGE_PREFIX"] = "gs://edgar_666/tmp"
    lock_path = "some_random_lock.json"
    assert write_lock(lock_path)
    assert not write_lock(lock_path)
    assert _get_lock_blob(lock_path).exists()  # pyright: ignore
    delete_lock(lock_path)
    assert not _get_lock_blob(lock_path).exists()  # pyright: ignore


def _chunk_and_get_embeddings(cik: str, accession_number: str, chunk_algo_version: str):
    filing = SECFiling(
        cik=cik,
        accession_number=accession_number,
    )
    filing_path, filing_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

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
    return chunks
