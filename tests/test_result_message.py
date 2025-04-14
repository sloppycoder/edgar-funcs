import os

import pytest  # noqa F401

from cli import _batch_id
from func_helpers import publish_message


@pytest.mark.skip(reason="for local use only")
def send_test_extraction_result():
    extraction_result = {
        "batch_id": _batch_id(),
        "cik": "1",
        "company_name": "test_company",
        "accession_number": "1",
        "date_filed": "2022-12-01",
        "selected_chunks": [123, 456],
        "selected_text": "some_text",
        "response": "{}",
        "notes": "some_notes",
        "model": "gemini-flash-2.0",
        "extraction_type": "trustee_comp",
    }
    publish_message(extraction_result, os.environ.get("EXTRACTION_RESULT_TOPIC", ""))
