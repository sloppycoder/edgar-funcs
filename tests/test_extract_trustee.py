import json
from functools import partial
from unittest.mock import patch

import pytest

from edgar_funcs.rag.extract.trustee import extract_trustee_comp_from_filing
from tests.utils import mock_file_content, mockdata_path

extract_func = partial(
    extract_trustee_comp_from_filing,
    embedding_model="vertex_ai/text-embedding-005",
    embedding_dimension=768,
    chunk_algo_version="3",
)


@pytest.mark.parametrize(
    "extraction_model", ["vertex_ai/gemini-2.0-flash-001", "gpt-4o-mini"]
)
@patch("edgar_funcs.rag.extract.trustee.ask_model")
def test_extract_html_filing(mock_ask_model, extraction_model):
    model_path = extraction_model.replace("/", "_")
    mock_ask_model.return_value = mock_file_content(
        f"response/{model_path}/1002427/0001133228-24-004879_trustee_comp.txt"
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


@pytest.mark.parametrize(
    "extraction_model", ["gpt-4o-mini", "vertex_ai/gemini-2.0-flash-001"]
)
@patch("edgar_funcs.rag.extract.trustee.ask_model")
def test_extract_txt_filing(mock_ask_model, extraction_model):
    model_path = extraction_model.replace("/", "_")
    mock_ask_model.return_value = mock_file_content(
        f"response/{model_path}/1201932/0000950136-04-001365_trustee_comp.txt"
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


@pytest.mark.parametrize(
    "extraction_model", ["vertex_ai/gemini-2.0-flash-001", "gpt-4o-mini"]
)
# @pytest.mark.skip(
#     reason="""
#     run this only for updating mock response,
#     after changing the prompt or parameters to completion API
#     """
# )
def test_update_mock_response_html_filing(extraction_model):
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
            / "0001133228-24-004879_trustee_comp.txt"
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


@pytest.mark.parametrize(
    "extraction_model", ["gpt-4o-mini", "vertex_ai/gemini-2.0-flash-001"]
)
# @pytest.mark.skip(
#     reason="""
#     run this only for updating mock response,
#     after changing the prompt or parameters to completion API
#     """
# )
def test_update_mock_response_txt_filing(extraction_model):
    result = extract_func(
        cik="1201932",
        accession_number="0000950136-04-001365",
        model=extraction_model,
    )

    if result and result.get("response"):
        model_path = extraction_model.replace("/", "_")
        response_file = (
            mockdata_path
            / "response"
            / model_path
            / "1201932"
            / "0000950136-04-001365_trustee_comp.txt"
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
