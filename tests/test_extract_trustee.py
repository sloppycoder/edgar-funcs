from functools import partial
from unittest.mock import patch

import pytest

from edgar_funcs.rag.extract.trustee import extract_trustee_comp_from_filing
from tests.utils import mock_file_content

extract_func = partial(
    extract_trustee_comp_from_filing,
    embedding_model="text-embedding-005",
    embedding_dimension=768,
    chunk_algo_version="3",
)


@pytest.mark.parametrize("extraction_model", ["gemini-2.0-flash"])
@patch("edgar_funcs.rag.extract.trustee.ask_model")
def test_extract_html_filing(mock_ask_model, extraction_model):
    mock_ask_model.return_value = mock_file_content(
        f"response/{extraction_model}/1002427/0001133228-24-004879_trustee_comp.txt"
    )
    result = extract_func(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=extraction_model,
    )
    assert (
        result
        and result["n_trustee"] == 11
        and result["comp_info"]["trustees"][0]["name"] == "Frank L. Bowman"
        and result["comp_info"]["trustees"][0]["fund_group_compensation"] == "400000"
    )


@pytest.mark.parametrize("extraction_model", ["gpt-4o-mini", "gemini-2.0-flash"])
@patch("edgar_funcs.rag.extract.trustee.ask_model")
def test_extract_txt_filing(mock_ask_model, extraction_model):
    mock_ask_model.return_value = mock_file_content(
        f"response/{extraction_model}/1201932/0000950136-04-001365_trustee_comp.txt"
    )
    result = extract_func(
        cik="1201932",
        accession_number="0000950136-04-001365",
        model=extraction_model,
    )
    assert (
        result
        and result["n_trustee"] == 9
        and result["comp_info"]["trustees"][0]["name"] == "Michael Bozic"
        and result["comp_info"]["trustees"][0]["fund_compensation"] == "668"
    )
