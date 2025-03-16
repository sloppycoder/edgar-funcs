import os
import sys
from pathlib import Path

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{cwd}/.."))

os.environ["STORAGE_PREFIX"] = str(Path(__file__).parent.parent / "tmp")
# os.environ["STORAGE_PREFIX"] = "gs://edgar_666/new_funcs"
