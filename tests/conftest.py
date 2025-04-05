import os
import sys
from pathlib import Path

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{cwd}/.."))

os.environ["STORAGE_PREFIX"] = str(Path(__file__).parent / "mockdata/pickle")
# os.environ["STORAGE_PREFIX"] = "gs://edgar_666/new_funcs"

# set models for testing
os.environ["EMBEDDING_MODEL"] = "text-embedding-005"
os.environ["EMBEDDING_DIMENSION"] = "768"
os.environ["EXTRACTION_MODEL"] = "gemini-1.5-flash-002"
# text-embedding-3-small
# gemini-2.0-flash
# gpt-4o-mini
