import json
from pathlib import Path

mockdata_path = Path(__file__).parent / "mockdata"


def mock_file_content(path: str, is_binary: bool = False):
    with open(mockdata_path / path, "rb" if is_binary else "r") as f:
        return f.read()


def mock_json_dict(path):
    with open(mockdata_path / path, "r") as f:
        return json.load(f)
