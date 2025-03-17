from unittest.mock import patch

from rag import extract_trustee_comp, load_chunks
from rag.embedding import GEMINI_EMBEDDING_MODEL
from tests.utils import mock_file_content


def test_extract_filing():
    chunks = load_chunks(
        cik="1002427",
        accession_number="0001133228-24-004879",
        model=GEMINI_EMBEDDING_MODEL,
        dimension=768,
    )
    queries = load_chunks(
        cik="0",
        accession_number="0",
        model=GEMINI_EMBEDDING_MODEL,
        dimension=768,
    )

    with patch(
        "rag.ask_model",
        return_value=mock_file_content(
            "response/gemini-1.5-flash-002/1002427/0001133228-24-004879.txt"
        ),
    ):
        result = extract_trustee_comp(queries, chunks)
        assert (
            result
            and result["n_trustee"] == 11
            and result["comp_info"]["trustees"][0]["name"] == "Frank L. Bowman"
            and result["comp_info"]["trustees"][0]["fund_group_compensation"] == "400000"
        )
