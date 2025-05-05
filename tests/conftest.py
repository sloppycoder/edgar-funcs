import os
from pathlib import Path

# comment the following out to use values set in .env
os.environ["STORAGE_PREFIX"] = str(Path(__file__).parent / "mockdata/pickle")
