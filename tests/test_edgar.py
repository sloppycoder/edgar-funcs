from unittest.mock import patch

import pytest  # noqa F401

from edgar_funcs.edgar import (
    SECFiling,
    _index_html_path,
    parse_idx_filename,
)
from tests.utils import mock_file_content


def test_idx_filename2index_html_path():
    assert (
        _index_html_path("edgar/data/1035018/0001193125-20-000327.txt")
        == "edgar/data/1035018/000119312520000327/0001193125-20-000327-index.html"
    )


def test_parse_idx_filename():
    assert ("1035018", "0001193125-20-000327") == parse_idx_filename(
        "edgar/data/1035018/0001193125-20-000327.txt"
    )
    with pytest.raises(ValueError, match="an unexpected format"):
        parse_idx_filename("edgar/data/blah.txt")


@patch("edgar_funcs.edgar.edgar_file")
def test_parse_485bpos_filing(mock_edgar_file):
    mock_edgar_file.side_effect = mock_file_content
    filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
    html_path, html_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

    assert filing.cik == "1002427" and filing.date_filed == "2024-04-29"
    assert filing.accession_number == "0001133228-24-004879"
    assert len(filing.documents) == 26
    assert html_path.endswith("msif-html7854_485bpos.htm")
    assert html_content and "N-1A" in html_content


@patch("edgar_funcs.edgar.edgar_file")
def test_parse_old_485bpos_filing(mock_edgar_file):
    mock_edgar_file.side_effect = mock_file_content
    filing = SECFiling(
        cik="1201932",
        accession_number="0000950136-04-001365",
        prefer_index_headers=False,
    )
    html_path, html_content = filing.get_doc_content(
        "485BPOS", file_types=["htm", "txt"]
    )[0]

    assert filing.cik == "1201932" and filing.date_filed == "2004-04-30"
    assert filing.accession_number == "0000950136-04-001365"
    assert len(filing.documents) == 9
    assert html_path.endswith("file001.txt")
    assert html_content and "N-1A" in html_content
