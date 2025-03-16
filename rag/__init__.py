import logging
from datetime import datetime
from typing import Any

from edgar import SECFiling

from .chunking import chunk_text, trim_html_content
from .embedding import GEMINI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL, batch_embedding

logger = logging.getLogger(__name__)


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
