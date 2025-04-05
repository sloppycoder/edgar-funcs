from unittest.mock import patch

from func_helpers import model_settings
from rag.extract.trustee import extract_trustee_comp_from_filing
from rag.vectorize.chunking import CHUNK_ALORITHM_VERSION
from tests.utils import mock_file_content

embedding_model, embedding_dimension, extraction_model = model_settings()


def test_extract_html_filing():
    with patch(
        "rag.extract.trustee.ask_model",
        return_value=mock_file_content(
            "response/gemini-1.5-flash-002/1002427/0001133228-24-004879_trustee_comp.txt"
        ),
    ):
        result = extract_trustee_comp_from_filing(
            cik="1002427",
            accession_number="0001133228-24-004879",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            model=extraction_model,
            chunk_algo_version=CHUNK_ALORITHM_VERSION,
        )
        assert (
            result
            and result["n_trustee"] == 11
            and result["comp_info"]["trustees"][0]["name"] == "Frank L. Bowman"
            and result["comp_info"]["trustees"][0]["fund_group_compensation"] == "400000"
        )


def test_extract_txt_filing():
    with patch(
        "rag.extract.trustee.ask_model",
        return_value=mock_file_content(
            "response/gemini-1.5-flash-002/1201932/0000950136-04-001365.txt"
        ),
    ):
        result = extract_trustee_comp_from_filing(
            cik="1201932",
            accession_number="0000950136-04-001365",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            model=extraction_model,
            chunk_algo_version=CHUNK_ALORITHM_VERSION,
        )
        assert (
            result
            and result["n_trustee"] == 9
            and result["comp_info"]["trustees"][0]["name"] == "Michael Bozic"
            and result["comp_info"]["trustees"][0]["fund_compensation"] == "668"
        )
