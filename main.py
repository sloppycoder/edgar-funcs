import logging
import os

import functions_framework
from google.cloud import logging as cloud_logging

from edgar import SECFiling
from rag import (
    DEFAULT_LLM_MODEL,
    TextChunksWithEmbedding,
    TrusteeComp,
    chunk_filing,
    extract_trustee_comp,
    load_chunks,
)
from rag.embedding import GEMINI_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def setup_cloud_logging():
    # Only initialize Google Cloud Logging in Cloud Run
    if os.getenv("K_SERVICE"):
        client = cloud_logging.Client()
        client.setup_logging()
        logging.info("Google Cloud Logging is enabled.")
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info("Using local logging.")


setup_cloud_logging()
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def edgar_processor(cloud_event):
    logger.info(f"edgar_processor received {cloud_event}")

    attrs = cloud_event.attributes
    event_type = attrs.get("type")
    data = cloud_event.data

    if event_type == "edgar.funcs.chunk":
        chunk_filing_and_save_embedding(data)
    elif event_type == "edgar.funcs.extract":
        extract_filing(data)
    else:
        # don't know how to process an event
        # it's probably meant for some other processors
        pass


def chunk_filing_and_save_embedding(data):
    cik = data.get("cik")
    accession_number = data.get("accession_number")
    model = data.get("model")
    dimension = data.get("dimension", 768)

    filing = SECFiling(cik=cik, accession_number=accession_number)
    text_chunks = chunk_filing(filing)

    model = model or GEMINI_EMBEDDING_MODEL
    metadata = {
        "cik": filing.cik,
        "accession_number": filing.accession_number,
        "date_filed": filing.date_filed,
        "model": model,
        "dimension": dimension,
    }
    filing_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
    filing_chunks.get_embeddings()
    return filing_chunks


def extract_filing(data) -> TrusteeComp | None:
    cik = data.get("cik")
    accession_number = data.get("accession_number")
    embedding_model = data.get("embedding_model")
    dimension = data.get("dimension", 768)
    model = data.get("model", DEFAULT_LLM_MODEL)

    chunks = load_chunks(
        cik=cik,
        accession_number=accession_number,
        model=embedding_model,
        dimension=dimension,
    )
    queries = load_chunks(
        cik=cik,
        accession_number=accession_number,
        model=GEMINI_EMBEDDING_MODEL,
        dimension=768,
    )

    if chunks is None or queries is None:
        return None

    result = extract_trustee_comp(queries, chunks, model)
    return result


if __name__ == "__main__":
    setup_cloud_logging()
