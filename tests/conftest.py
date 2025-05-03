import os
import sys
from pathlib import Path

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(f"{cwd}/.."))

# comment the following out to use values set in .env
os.environ["STORAGE_PREFIX"] = str(Path(__file__).parent / "mockdata/pickle")
