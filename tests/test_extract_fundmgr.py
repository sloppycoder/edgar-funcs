from functools import partial
from unittest.mock import patch

import pytest  # noqa: F401

from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from tests.utils import mock_file_content

extract_func = partial(
    extract_fundmgr_ownership_from_filing,
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
    chunk_algo_version="4",
)


@pytest.mark.parametrize("extraction_model", ["gpt-4o-mini", "gemini-2.0-flash"])
@patch("edgar_funcs.rag.extract.fundmgr.ask_model")
def test_extract_fundmgr_ownership(mock_ask_model, extraction_model):
    mock_ask_model.return_value = mock_file_content(
        f"response/{extraction_model}/1002427/0001133228-24-004879_fundmgr_ownership.txt"
    )
    result = extract_func(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=extraction_model,
    )
    assert result and result["ownership_info"]
    managers = result["ownership_info"]["managers"]
    assert len(managers) == 6
    assert (
        managers[0]["name"] == "Dennis P. Lynch"
        and managers[0]["ownership_range"].replace(",", "") == "Over 1000000"
    )
