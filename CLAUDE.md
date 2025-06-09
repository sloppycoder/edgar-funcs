# CLAUDE.md

## Project Overview
This is a Google Cloud Functions project that processes SEC filings from the EDGAR site. The system uses RAG (Retrieval Augmented Generation) to extract information from 485BPOS filings by chunking documents and generating embeddings for LLM-based extraction.

## Architecture
- **Cloud Run service** for processing requests
- **Pub/Sub topics** for request/response messaging
- **BigQuery** for storing extraction results
- **CLI utility** for submitting processing requests

## Key Dependencies
- Python 3.12+
- UV package manager (uv.lock present)
- Google Cloud Platform services (Pub/Sub, BigQuery, Cloud Run, AI Platform)
- LLM providers: OpenAI and Google Vertex AI
- Key libraries: spacy, beautifulsoup4, pandas, litellm, tiktoken

## Development Setup
```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev

# Install notebook dependencies (optional)
uv sync --group notebook
```

## Environment Variables
Create a `.env` file with:
```env
PUBSUB_TOPIC=edgarai-request
RESULT_TABLE=<project>.<dataset>.extraction_result
GOOGLE_APPLICATION_CREDENTIALS=<service_account_credentials>.json
CLI_EDGAR_PROCESSOR_URL=https://YOUR_CLOUD_RUN_URL/process
EXTRACTION_RESULT_TOPIC=<result_topic_name>
```

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
pytest tests/test_main.py
```

### Code Quality
```bash
# Run linting
ruff check .

# Run linting with auto-fix
ruff check . --fix

# Type checking
pyright
```

### CLI Usage
```bash
# Process a single filing by accession number
python cli.py chunk 0000000000-00-000000

# Process filings for 10% of companies in date range
python cli.py chunk 10 --start 2024-01-01 --end 2024-12-31

# Process filings from CSV file
python cli.py chunk filings.csv

# Extract trustee information
python cli.py trustee 0000000000-00-000000

# Extract fund manager information
python cli.py fundmgr 0000000000-00-000000

# Get statistics for batch
python cli.py stats 20240101000000-abc

# Export results
python cli.py export 20240101000000-abc
```

### Notebook Development
```bash
# Start Marimo notebook environment
marimo edit notebooks
```

### Deployment
```bash
# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Set up Pub/Sub topics and subscriptions
./scripts/req_topic.sh
./scripts/result_topic.sh
```

## Project Structure
- `edgar_funcs/` - Main package
  - `edgar.py` - SEC filing data handling
  - `rag/` - RAG implementation
    - `extract/` - LLM extraction modules (algo, fundmgr, trustee)
    - `vectorize/` - Text chunking and embedding
- `cli.py` - Command-line interface
- `main.py` - Flask app for Cloud Run
- `func_helpers.py` - Utility functions
- `tests/` - Test suite
- `scripts/` - Deployment and setup scripts
- `notebooks/` - Analysis notebooks

## Key Files
- `pyproject.toml` - Project configuration and dependencies
- `pyrightconfig.json` - Type checker configuration
- `requirements.txt` - Auto-generated dependency list
- `Dockerfile` - Container configuration
- `cloudbuild.yaml` - Cloud Build configuration

## Testing Strategy
The project includes comprehensive tests covering:
- CLI functionality (`test_cli.py`)
- Main processing logic (`test_main.py`)
- EDGAR data handling (`test_edgar.py`)
- Text chunking (`test_chunking.py`)
- Embedding generation (`test_embedding.py`)
- Extraction algorithms (`test_extract_*.py`)

## Embedding Models
Default: `text-embedding-3-small` (OpenAI, 1536 dimensions)
Alternative: `text-embedding-005` (Google, 768 dimensions)

## LLM Models
Default: `vertex_ai/gemini-2.0-flash-001`
Alternative: `gpt-4o-mini`

## Claude Code Instructions

**IMPORTANT**: When working on this project, Claude Code must ALWAYS follow these steps after making any code changes:

1. **Run Ruff linting and auto-fix** after every code modification:
   ```bash
   ruff check . --fix
   ```

2. **Check for remaining linting issues**:
   ```bash
   ruff check .
   ```

3. **Project-specific linting rules**:
   - Line length limit: **90 characters** (configured in pyproject.toml)
   - Indent width: **4 spaces**
   - Always fix simple issues like line length, imports, spacing automatically
   - Follow PEP8 standards and project conventions

4. **After fixing linting issues, run tests** to ensure nothing is broken:
   ```bash
   pytest
   ```

5. **Type checking** (optional but recommended):
   ```bash
   pyright
   ```

**Never skip the ruff auto-fix step** - it's configured to handle most formatting issues automatically, including line length violations, import sorting, and spacing issues.
