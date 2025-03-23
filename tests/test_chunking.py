from unittest.mock import patch

from edgar import SECFiling
from rag.vectorize import TextChunksWithEmbedding
from rag.vectorize.chunking import _is_line_empty, chunk_text, trim_html_content
from rag.vectorize.embedding import GEMINI_EMBEDDING_MODEL
from tests.utils import mock_file_content


def test_chunk_html_filing():
    # newer filing usually uses an -index-headers.html for filing content
    # and has html for the filing itself
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
        filing_path, filing_content = filing.get_doc_content("485BPOS", max_items=1)[0]

        assert filing_path.endswith(".html") or filing_path.endswith(".htm")

        trimmed_html = trim_html_content(filing_content)
        chunks = chunk_text(trimmed_html, method="spacy")

        assert len(chunks) == 262
        # no chunk is empty or too short
        assert all(chunk and len(chunk) > 10 for chunk in chunks)


def test_chunk_txt_filing():
    # older filing does not have an -index-headers.html for filing content
    # and the filing content can be in a txt file not html
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(
            cik="39473",
            accession_number="0000039473-03-000002",
            prefer_index_headers=False,
        )
        filing_path, filing_content = filing.get_doc_content("485BPOS", max_items=1)[0]

        assert filing_path.endswith(".txt")

        chunks = chunk_text(filing_content, method="spacy")

        assert len(chunks) == 177
        # no chunk is empty or too short
        assert all(chunk and len(chunk) > 10 for chunk in chunks)

        # uncomment to save chunks pickle file to mockdata folder
        # _save_chunks_mockdata(filing, chunks)


def test_chunk_filing_with_difficult_table():
    # this filing has a weird table that was causing issues
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="1274676", accession_number="0000919574-24-004904")
        filing_path, filing_content = filing.get_doc_content("485BPOS", max_items=1)[0]

        assert filing_path.endswith(".html") or filing_path.endswith(".htm")

        trimmed_html = trim_html_content(filing_content)
        chunks = chunk_text(trimmed_html, method="spacy")

        assert len(chunks) == 208
        # no chunk is empty or too short
        assert all(chunk and len(chunk) > 10 for chunk in chunks)

        chunk_of_table = [chunk for chunk in chunks if "2,691" in chunk][0]
        assert (
            "Name of Trustee |  Aggregate Compensation" in chunk_of_table
        )  # beginning of the table
        assert "Emilie D. Wrapp" in chunk_of_table  # end of the table


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
    new_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
    new_chunks.get_embeddings(model=GEMINI_EMBEDDING_MODEL, dimension=768)
    new_chunks.save()
