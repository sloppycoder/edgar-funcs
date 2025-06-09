import json
from functools import partial
from unittest.mock import patch

import pytest  # noqa: F401

from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from tests.utils import mock_file_content, mockdata_path

extract_func = partial(
    extract_fundmgr_ownership_from_filing,
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
    chunk_algo_version="4",
)


@pytest.mark.parametrize(
    "extraction_model", ["gpt-4o-mini", "vertex_ai/gemini-2.0-flash-001"]
)
@patch("edgar_funcs.rag.extract.fundmgr.ask_model")
def test_extract_fundmgr_ownership(mock_ask_model, extraction_model):
    model_path = extraction_model.replace("/", "_")
    mock_ask_model.return_value = mock_file_content(
        f"response/{model_path}/1002427/0001133228-24-004879_fundmgr_ownership.txt"
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


@pytest.mark.parametrize(
    "extraction_model", ["gpt-4o-mini", "vertex_ai/gemini-2.0-flash-001"]
)
@pytest.mark.skip(
    reason="""
    run this only for updating mock response,
    after changing the prompt or parameters to completion API
    """
)
def test_update_mock_response(extraction_model):
    result = extract_func(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=extraction_model,
    )

    if result and result.get("response"):
        model_path = extraction_model.replace("/", "_")
        response_file = (
            mockdata_path
            / "response"
            / model_path
            / "1002427"
            / "0001133228-24-004879_fundmgr_ownership.txt"
        )

        response_file.parent.mkdir(parents=True, exist_ok=True)

        response = result["response"]
        assert response is not None, "Response should not be None"

        try:
            response_json = json.loads(response)
            formatted_response = json.dumps(response_json, indent=2)
        except json.JSONDecodeError:
            formatted_response = response

        with open(response_file, "w") as f:
            f.write(formatted_response)
