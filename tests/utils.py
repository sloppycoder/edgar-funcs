import json
from pathlib import Path

mockdata_path = Path(__file__).parent / "mockdata"


def mock_file_content(path):
    with open(mockdata_path / path, "r") as f:
        return f.read()


def mock_embedding(path):
    with open(mockdata_path / "embeddings" / path, "r") as f:
        return json.load(f)
