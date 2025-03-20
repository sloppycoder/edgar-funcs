from unittest.mock import patch

from rag.extract.llm import DEFAULT_LLM_MODEL
from rag.extract.trustee import extract_filing
from rag.vectorize.embedding import GEMINI_EMBEDDING_MODEL
from tests.utils import mock_file_content


def test_extract_filing():
    with patch(
        "rag.extract.trustee.ask_model",
        return_value=mock_file_content(
            "response/gemini-1.5-flash-002/1002427/0001133228-24-004879.txt"
        ),
    ):
        result = extract_filing(
            cik="1002427",
            accession_number="0001133228-24-004879",
            embedding_model=GEMINI_EMBEDDING_MODEL,
            embedding_dimension=768,
            model=DEFAULT_LLM_MODEL,
        )
        assert (
            result
            and result["n_trustee"] == 11
            and result["comp_info"]["trustees"][0]["name"] == "Frank L. Bowman"
            and result["comp_info"]["trustees"][0]["fund_group_compensation"] == "400000"
        )
