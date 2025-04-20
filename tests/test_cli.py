import shlex
from unittest.mock import ANY, patch

import pytest

from cli import main
from edgar_funcs.edgar import load_filing_catalog


@pytest.fixture
def mock_publish_request():
    with patch("cli._publish_message") as mock:
        yield mock


@pytest.fixture
def mock_send_cloud_run_request():
    with patch("cli.send_cloud_run_request") as mock:
        yield mock


def test_chunk_accession_number(
    mock_publish_request,
    mock_send_cloud_run_request,
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        shlex.split("cli.py chunk 0001224568-24-000005"),
    )
    main()
    assert mock_publish_request.call_count == 0
    assert mock_send_cloud_run_request.call_count == 1
    mock_send_cloud_run_request.assert_called_once_with(
        ANY,
        {
            "batch_id": "single",
            "action": "chunk",
            "cik": "1224568",
            "company_name": ANY,
            "accession_number": "0001224568-24-000005",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "model": "gemini-2.0-flash",
            "chunk_algo_version": "4",
        },
    )


def test_trustee_sample(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", shlex.split("cli.py trustee 1 --start 2024-01-01 --end 2024-12-31")
    )
    main()
    assert mock_publish_request.call_count >= 13 and mock_publish_request.call_count <= 15
    for call_args in mock_publish_request.call_args_list:
        assert call_args[0][0]["action"] == "trustee"


def test_fundmgr_list(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        shlex.split(
            "cli.py fundmgr tests/mockdata/cli/filing_list.csv --start 2024-01-01 --end 2024-12-31"  # noqa E501
        ),
    )
    main()
    assert mock_publish_request.call_count == 4
    for call_args in mock_publish_request.call_args_list:
        assert call_args[0][0]["action"] == "fundmgr"


def test_invalid_accession_number(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        # accession number has the wrong format
        shlex.split("cli.py chunk 0000000000-00 --start 2024-01-01 --end 2024-12-31"),
    )
    with pytest.raises(SystemExit):
        main()
    mock_publish_request.assert_not_called()


def test_print_stats(monkeypatch):
    monkeypatch.setattr("sys.argv", shlex.split("cli.py stats 20250420142007-lgh"))
    with patch("cli.print_stats") as mock:
        main()
    mock.assert_called_once_with("20250420142007-lgh")


def test_non_existent_accession_number(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        # accession number has the right format but doesn't match any filing
        shlex.split(
            "cli.py chunk 0000000000-00-000000 --start 2024-01-01 --end 2024-12-31"
        ),
    )
    main()
    mock_publish_request.assert_not_called()


def test_load_catalog():
    df_filings = load_filing_catalog("2000-01-01", "2024-12-31")
    cik1_filings = df_filings[df_filings["cik"] == "1342947"]
    assert len(cik1_filings) > 0
