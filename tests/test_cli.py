import shlex
from unittest.mock import ANY, patch

import pytest

from cli import main
from edgar_funcs.edgar import load_filing_catalog


@pytest.fixture
def mock_publish_request():
    with patch("cli.publish_message") as mock:
        yield mock


def test_chunk_command(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", shlex.split("cli.py chunk 1 --start 2024-01-01 --end 2024-12-31")
    )
    main()
    assert mock_publish_request.call_count > 0
    for call_args in mock_publish_request.call_args_list:
        assert call_args[0][0]["action"] == "chunk"


def test_trustee_command(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", shlex.split("cli.py trustee 50 --start 2024-01-01 --end 2024-12-31")
    )
    main()
    assert mock_publish_request.call_count > 0
    for call_args in mock_publish_request.call_args_list:
        assert call_args[0][0]["action"] == "trustee"


def test_fundmgr_command(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", shlex.split("cli.py fundmgr 50 --start 2024-01-01 --end 2024-12-31")
    )
    main()
    assert mock_publish_request.call_count > 0
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


def test_specific_accession_number(mock_publish_request, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        shlex.split("cli.py chunk 0001224568-24-000005"),
    )
    main()
    mock_publish_request.assert_called_once_with(
        {
            "batch_id": ANY,
            "action": "chunk",
            "cik": "1224568",
            "company_name": ANY,
            "accession_number": "0001224568-24-000005",
            "embedding_model": "text-embedding-005",
            "embedding_dimension": 768,
            "model": "gemini-2.0-flash",
            "chunk_algo_version": "4",
        },
        ANY,
    )


def test_load_catalog():
    df_filings = load_filing_catalog("2000-01-01", "2024-12-31")
    cik1_filings = df_filings[df_filings["cik"] == "1342947"]
    assert len(cik1_filings) > 0
