import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from ...edgar import SECFiling
from ..helper import gcs_client
from .chunking import chunk_text, trim_html_content
from .embedding import batch_embedding

logger = logging.getLogger(__name__)


class TextEmbeddingMetadata(TypedDict):
    cik: str
    accession_number: str
    date_filed: str
    model: str
    dimension: int
    chunk_algo_version: str


class TextChunksWithEmbedding:
    texts: list[str]
    embeddings: list[list[float]]
    metadata: TextEmbeddingMetadata

    def __init__(
        self,
        texts: list[str],
        embeddings: list[list[float]] = [],
        metadata={
            "accession_number": "",
            "date_filed": "",
            "model": "",
            "dimension": 0,
            "chunk_algo_version": "0",
        },
    ):
        if not texts:
            raise ValueError("texts cannot be empty")

        self.texts = texts
        self.embeddings = embeddings
        self.metadata = metadata

    def is_ready(self) -> bool:
        return (
            self.metadata["model"] != ""
            and self.metadata["dimension"] > 0
            and len(self.texts) > 0
            and len(self.embeddings) > 0
            and len(self.texts) == len(self.embeddings)
            and len(self.embeddings[0]) == self.metadata["dimension"]
        )

    def get_embeddings(
        self,
        model: str,
        dimension: int,
    ):
        if not self.texts:
            raise ValueError("texts cannot be empty")

        start_t = datetime.now()
        self.embeddings = batch_embedding(self.texts, model=model, dimension=dimension)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"batch_embedding of {len(self.texts)} chunks of text with {model} took {elapsed_t.total_seconds()} seconds"  # noqa E501
        )
        self.metadata["model"] = model
        self.metadata["dimension"] = dimension

    def get_text_chunks(self, chunks: list[int]) -> str:
        return "\n\n".join([self.texts[i] for i in chunks])

    def save(self) -> None:
        if self.is_ready():
            path = _blob_path(**self.metadata)
            bucket_name, prefix = _storage_prefix()
            obj = [self.texts, self.embeddings, self.metadata]
            if bucket_name:
                # use GCS bucket
                bucket = gcs_client().bucket(bucket_name)
                blob = bucket.blob(f"{prefix}/{path}")
                blob.upload_from_string(pickle.dumps(obj))
            else:
                # save to local file system
                output_path = Path(prefix) / path
                os.makedirs(output_path.parent, exist_ok=True)
                with open(output_path, "wb") as f:
                    pickle.dump(obj, f)
        else:
            raise ValueError("cannot save without embedding data")

    @classmethod
    def load(
        cls,
        cik: str,
        accession_number: str,
        model: str,
        dimension: int,
        chunk_algo_version: str,
    ) -> "TextChunksWithEmbedding":
        path = _blob_path(
            cik=cik,
            accession_number=accession_number,
            model=model,
            dimension=dimension,
            chunk_algo_version=chunk_algo_version,
        )
        bucket_name, prefix = _storage_prefix()
        obj = None
        if bucket_name:
            # use GCS bucket
            bucket = gcs_client().bucket(bucket_name)
            blob = bucket.blob(f"{prefix}/{path}")
            if blob.exists():
                obj = pickle.loads(blob.download_as_bytes())
        else:
            # load from local file system
            output_path = Path(prefix) / path
            if output_path.exists():
                with open(output_path, "rb") as f:
                    obj = pickle.load(f)

        if obj and len(obj) == 3:
            return TextChunksWithEmbedding(
                texts=obj[0], embeddings=obj[1], metadata=obj[2]
            )

        raise ValueError(f"Cannot load chunks from {path}")


def chunk_filing(filing: SECFiling, method: str = "spacy") -> list[str]:
    if method != "spacy":
        raise ValueError(f"Unsupported chunking method {method}")

    path, content = filing.get_doc_content("485BPOS", max_items=1)[0]
    text_content = None

    if path.endswith((".html", ".htm")):
        text_content = trim_html_content(content)
    elif path.endswith(".txt"):
        text_content = content
    else:
        raise ValueError(f"Unsupported document format {path}")

    start_t = datetime.now()
    chunks = chunk_text(text_content, method=method)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"chunking {filing.cik}/{filing.accession_number} into {len(chunks)} chunks took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )
    return chunks


def chunk_filing_and_save_embedding(
    cik: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_algo_version: str,
    refresh: bool = False,
    **_,  # ignore any other parameters
) -> tuple[bool, TextChunksWithEmbedding]:
    try:
        if refresh:
            raise ValueError("refresh is True")

        existing_chunks = TextChunksWithEmbedding.load(
            cik=cik,
            accession_number=accession_number,
            model=embedding_model,
            dimension=embedding_dimension,
            chunk_algo_version=chunk_algo_version,
        )
        return True, existing_chunks
    except ValueError:
        filing = SECFiling(cik=cik, accession_number=accession_number)
        text_chunks = chunk_filing(filing)
        metadata = {
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
            "chunk_algo_version": chunk_algo_version,
        }
        new_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
        new_chunks.get_embeddings(model=embedding_model, dimension=embedding_dimension)
        new_chunks.save()
        return False, new_chunks


def _blob_path(
    cik: str,
    accession_number: str,
    model: str,
    dimension: int,
    chunk_algo_version: str,
    **_,  # ignore any other parameters
):
    # return the path for storing a TextChunkWithEmbedding object
    return (
        f"chunks/{chunk_algo_version}/{model}_{dimension}/{cik}/{accession_number}.pickle"
    )


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
