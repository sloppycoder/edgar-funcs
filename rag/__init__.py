import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from google.cloud import storage

from edgar import SECFiling

from .chunking import ALORITHM_VERSION, chunk_text, trim_html_content
from .embedding import GEMINI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL, batch_embedding

logger = logging.getLogger(__name__)

gcs_client = storage.Client()


class TextChunkWithEmbedding:
    model: str
    embbedding_dimensions: int = 0
    texts: list[str] = []
    embeddings: list[list[float]] = []
    metadata: dict[str, Any] = {}

    def __init__(self, texts: list[str], metadata: dict[str, Any] = {}):
        if not texts:
            raise ValueError("texts cannot be empty")

        self.texts = texts
        self.medata = metadata

    def is_ready(self) -> bool:
        return (
            len(self.texts) > 0
            and len(self.embeddings) > 0
            and len(self.texts) == len(self.embeddings)
            and len(self.embeddings[0]) == self.embbedding_dimensions
        )

    def get_embeddings(
        self,
        model: str = GEMINI_EMBEDDING_MODEL,
        dimension: int = 768,
    ):
        if model == OPENAI_EMBEDDING_MODEL:
            dimension = 1536

        start_t = datetime.now()
        self.embeddings = batch_embedding(self.texts, model=model, dimension=dimension)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"batch_embedding of {len(self.texts)} chunks of text with {model} took {elapsed_t.total_seconds()} seconds"  # noqa E501
        )
        self.embbedding_dimensions = dimension
        self.model = model


def save_chunks(chunks: TextChunkWithEmbedding) -> None:
    if chunks.is_ready():
        path = _blob_path(chunks)
        bucket_name, prefix = _storage_prefix()
        if bucket_name:
            # means it's GCS bucket
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(prefix + path)
            blob.upload_from_string(pickle.dumps(chunks))
        else:
            # save to local file system
            output_path = Path(prefix) / path
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(chunks, f)
    else:
        raise ValueError("cannot save without embedding data")


def load_chunks(
    cik: str,
    accession_number: str,
    model: str,
    dimension: int,
    chunk_version: str = ALORITHM_VERSION,
) -> TextChunkWithEmbedding:
    path = _blob_path(
        cik=cik,
        accession_number=accession_number,
        model=model,
        dimension=dimension,
        chunk_version=chunk_version,
    )
    bucket_name, prefix = _storage_prefix()
    if bucket_name:
        # means it's GCS bucket
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(prefix + path)
        chunks = pickle.loads(blob.download_as_bytes())
    else:
        # load from local file system
        output_path = Path(prefix) / path
        with open(output_path, "rb") as f:
            chunks = pickle.load(f)
    return chunks


def chunk_filing(filing: SECFiling, method: str = "spacy"):
    if method != "spacy":
        raise ValueError(f"Unsupported chunking method {method}")

    path, content = filing.get_doc_content("485BPOS", max_items=1)[0]
    if not path.endswith((".html", ".htm")):
        raise ValueError(f"Unsupported document format {path}")

    start_t = datetime.now()
    trimmed_html = trim_html_content(content)
    chunks = chunk_text(trimmed_html, method=method)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"chunking {filing.cik}/{filing.accession_number} into {len(chunks)} chunks took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )
    return chunks


def _blob_path(
    chunks: TextChunkWithEmbedding | None = None,
    cik: str = "",
    accession_number: str = "",
    chunk_version: str = "",
    model: str = "",
    dimension: int = 768,
):
    # return the path for storing a TextChunkWithEmbedding object
    if chunks and chunks.is_ready():
        # chunks object override rest of the parameters
        meta = chunks.medata
        cik = meta.get("cik") or "0"
        accession_number = meta.get("accession_number") or "0000000000-00-000000"
        chunk_version = meta.get("chunk_version") or ALORITHM_VERSION
        model = meta.get("model") or GEMINI_EMBEDDING_MODEL
        dimension = meta.get("dimension") or 768
    elif cik and accession_number and model and dimension:
        chunk_version = chunk_version or ALORITHM_VERSION
    else:
        raise ValueError("Must specifcy cik, accession_number, model, dimension")

    return f"{chunk_version}/{model}_{dimension}/{cik}/{accession_number}.pickle"


def _storage_prefix(storage_base_path=os.environ.get("STORAGE_PREFIX", "")):
    # return tuple of (bucket_name, prefix)
    # if the env var does not beginw with "gs://", returns ""
    if storage_base_path.startswith("gs://"):
        parts = storage_base_path[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    else:
        if storage_base_path.startswith("/"):
            return None, storage_base_path
        else:
            local_path = Path(__file__).parent.parent / storage_base_path
            return None, str(local_path)
