import base64
import json
from unittest.mock import patch

import pytest

from main import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


@patch("main._publish_result")
@patch("main._perform_extraction")
@patch("main._retrieve_chunks_for_filing")
@patch("main.mark_job_in_progress")
@patch("main.mark_job_done")
def test_req_processor_success(
    mock_mark_job_done,
    mock_mark_job_in_progress,
    mock_retrieve_chunks,
    mock_perform_extraction,
    mock_publish_result,
    client,
):
    mock_mark_job_in_progress.return_value = True
    mock_retrieve_chunks.return_value = type(
        "MockChunks", (), {"metadata": {"date_filed": "2023-01-01"}}
    )()
    mock_perform_extraction.return_value = {"response": "some giberish"}

    data = {
        "batch_id": "some-id",
        "action": "trustee",
        "cik": "1",
        "company_name": "company_name",
        "accession_number": "0",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "model": "gemini-2.0-flash",
        "chunk_algo_version": "4",
    }

    data_bytes = json.dumps(data).encode("utf-8")
    response = client.post(
        "/process",
        json={
            "message": {"data": base64.b64encode(data_bytes).decode("utf-8")},
        },
    )

    # Assertions
    assert response.status_code == 200
    assert response.json == {"response": "some giberish"}
    mock_publish_result.assert_called_once()
    mock_mark_job_done.assert_called_once()
