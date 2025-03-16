from unittest.mock import patch

import pytest

from edgar import (
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


def test_parse_485bpos_filing():
    with patch("edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="883622", accession_number="0001137439-24-001242")

        # filing = SECFiling(idx_filename="edgar/data/1002427/0001133228-24-004879.txt")
        html_path, html_content = filing.get_doc_content("485BPOS", max_items=1)[0]

        assert filing.cik == "883622" and filing.date_filed == "2024-08-14"
        assert filing.accession_number == "0001137439-24-001242"
        assert len(filing.documents) == 93
        assert html_path.endswith("ivyfunds08142024485bpos.htm")
        assert html_content and "N-1A" in html_content
