from unittest.mock import patch

from edgar import SECFiling
from rag.chunking import _is_line_empty, chunk_text, trim_html_content
from tests.utils import mock_file_content


def test_chunk_filing():
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="883622", accession_number="0001137439-24-001242")
        filing_path, filing_content = filing.get_doc_content("485BPOS", max_items=1)[0]

        assert filing_path.endswith(".html") or filing_path.endswith(".htm")

        trimmed_html = trim_html_content(filing_content)
        chunks = chunk_text(trimmed_html, method="spacy")

        assert len(chunks) == 9
        # no chunk is empty or too short
        assert all(chunk and len(chunk) > 10 for chunk in chunks)


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
