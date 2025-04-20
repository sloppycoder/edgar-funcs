import uuid

import pytest  # noqa F401

from cli import _batch_id
from func_helpers import mark_job_done, mark_job_in_progress
from main import _publish_result


@pytest.mark.skip(reason="for local use only")
def test_save_extraction_result():
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
        "model": "gemini-2.0-flash",
        "extraction_type": "trustee_comp",
    }
    _publish_result(extraction_result)


@pytest.mark.skip(reason="for local use only")
def test_mark_job_in_progress():
    job_id = str(uuid.uuid4())
    assert mark_job_in_progress(job_id)
    assert mark_job_in_progress(job_id) is False
    assert mark_job_done(job_id)
