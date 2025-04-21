from unittest.mock import patch

import pytest  # noqa: F401

from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
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
