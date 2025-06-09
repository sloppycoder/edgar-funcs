import json
from functools import partial

import pytest

from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from edgar_funcs.rag.extract.trustee import extract_trustee_comp_from_filing
from tests.utils import mockdata_path

update_fundmgr_mocks = False
update_trustee_mocks = False


# Setup extraction functions
extract_fundmgr_func = partial(
    extract_fundmgr_ownership_from_filing,
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
    chunk_algo_version="4",
)

extract_trustee_func = partial(
    extract_trustee_comp_from_filing,
    embedding_model="vertex_ai/text-embedding-005",
    embedding_dimension=768,
    chunk_algo_version="3",
)


@pytest.mark.skipif(
    not update_fundmgr_mocks,
    reason="only run for updating mock response file after changing parameters to ask_model",  # noqa: E501
)
@pytest.mark.parametrize(
    "extraction_model", ["gpt-4o-mini", "vertex_ai/gemini-2.0-flash-001"]
)
def test_update_mock_response_fundmgr(extraction_model):
    result = extract_fundmgr_func(
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


@pytest.mark.skipif(
    not update_trustee_mocks,
    reason="only run for updating mock response file after changing parameters to ask_model",  # noqa: E501
)
@pytest.mark.parametrize(
    "extraction_model", ["vertex_ai/gemini-2.0-flash-001", "gpt-4o-mini"]
)
def test_update_mock_response_trustee_html_filing(extraction_model):
    result = extract_trustee_func(
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


@pytest.mark.skipif(
    not update_trustee_mocks,
    reason="only run for updating mock response file after changing parameters to ask_model",  # noqa: E501
)
@pytest.mark.parametrize(
    "extraction_model", ["gpt-4o-mini", "vertex_ai/gemini-2.0-flash-001"]
)
def test_update_mock_response_trustee_txt_filing(extraction_model):
    result = extract_trustee_func(
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
