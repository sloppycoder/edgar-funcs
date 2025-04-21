import io
from unittest.mock import patch

import pandas as pd
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


def test_parse_485bpos_filing():
    with patch("edgar_funcs.edgar.edgar_file", side_effect=mock_file_content):
        filing = SECFiling(cik="1002427", accession_number="0001133228-24-004879")
        html_path, html_content = filing.get_doc_content(
            "485BPOS", file_types=["htm", "txt"]
        )[0]

        assert filing.cik == "1002427" and filing.date_filed == "2024-04-29"
        assert filing.accession_number == "0001133228-24-004879"
        assert len(filing.documents) == 26
        assert html_path.endswith("msif-html7854_485bpos.htm")
        assert html_content and "N-1A" in html_content


def test_parse_old_485bpos_filing():
    # this is an old filing where index-headers.html does not exist
    # so we must parse index.html to get the documents list
    with patch("edgar_funcs.edgar.edgar_file", side_effect=mock_file_content):
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


def test_load_filing_catalog():
    catalog_blob = mock_file_content(
        "../../edgar_funcs/data/catalog/all_485bpos_pd.pickle", is_binary=True
    )
    df_filings = pd.read_pickle(io.BytesIO(catalog_blob))
    assert df_filings.size > 17000

    # from pympler import asizeof
    # mem_used = asizeof.asizeof(df_filings) / 1024 / 1024
    # print(f"loaded catalog , used memory {mem_used:.2f} MB")


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
