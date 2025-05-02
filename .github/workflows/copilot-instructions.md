# Copilot Instructions

## Naming Conventions

### Files and Directories
- Use snake_case for Python filenames (e.g., `func_helpers.py`, `main.py`).
- Organize related functionality into directories (e.g., `edgar_funcs/rag/vectorize`).

### Variables and Functions
- Use snake_case for variable and function names (e.g., `decode_request`, `write_lock`).
- Prefix private helper functions with an underscore (e.g., `_retrieve_chunks_for_filing`).

### Classes
- Use PascalCase for class names (e.g., `SECFiling`, `TextChunksWithEmbedding`).

### Constants
- Use ALL_CAPS for constants (e.g., `DEFAULT_USER_AGENT`, `EDGAR_BASE_URL`).

## Code Organization

### tooling
- use `uv` as package mananger
- use `ruff` as code formatter
- use `pyright` as type checker

### Imports
- Group imports into standard library, third-party, and local imports.
- Maintain alphabetical order within each group.

### Functions
- Use type hints for function arguments and return values.
- Use docstrings to describe the purpose and behavior of functions.

### Logging
- Use the `logging` module for logging messages.
- Configure logging levels appropriately (e.g., `INFO`, `DEBUG`).

## Best Practices

### Environment Variables
- Use `os.environ` to access environment variables.
- Provide default values where necessary (e.g., `os.environ.get("GCP_REGION", "us-central1")`).

## Testing
- Place test files in the `tests/` directory.
- Use descriptive names for test files and functions (e.g., `test_chunking.py`, `test_extract_fundmgr.py`).

## Additional Notes
- Follow PEP 8 guidelines for Python code style.
- Use `dotenv` for managing environment variables in local development.
- Ensure all public-facing functions and APIs have clear and concise documentation.
