import base64
import json
import logging
import sys
import traceback
from typing import Any

import functions_framework
from cloudevents.http import CloudEvent

from edgar import SECFiling
from func_helpers import (
    publish_request,
    publish_response,
    setup_cloud_logging,
)
from rag import (
    DEFAULT_LLM_MODEL,
    TextChunksWithEmbedding,
    TrusteeComp,
    chunk_filing,
    extract_trustee_comp,
    load_chunks,
    save_chunks,
)
from rag.embedding import GEMINI_EMBEDDING_MODEL

setup_cloud_logging()
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def req_processor(cloud_event: CloudEvent):
    logger.info(f"req_processor received {cloud_event.data}")
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    try:
        if dispatch_event(data):
            publish_response({"params": data}, True, "success")
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.info(f"chunking with {data} failed with {error_msg}\n{tb}")
        publish_response({"params": data}, True, error_msg)


def dispatch_event(data: dict[str, Any]) -> Any:
    action = data.get("action")

    if action == "chunk_one_filing":
        is_reuse, chunks = chunk_filing_and_save_embedding(**data)
        if is_reuse:
            logger.info(f"re-use existing chunks for {data}")
        else:
            logger.info(f"created new chunks for {data}")
        return chunks
    elif action == "extract_one_filing":
        result = extract_filing(**data)
        if result:
            logger.info(f"extraction with {data} found {result['n_trustee']} trustees")
        return result
    else:
        logger.info(f"Unknown action {action}")
        return None


@functions_framework.cloud_event
def resp_processor(cloud_event: CloudEvent):
    logger.info(f"resp_processor received {cloud_event.data}")
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    action = data["action"]

    if action == "chunk_one_filing":
        req_is_success = data["success"]
        params = data["params"]
        if req_is_success and params.get("run_extract", False):
            params["action"] = "extract_one_filing"
            publish_request(params)
            logger.info(f"publish request for {params}")


def chunk_filing_and_save_embedding(
    cik: str,
    accession_number: str,
    embedding_model: str,
    dimension: int,
    refresh: bool = False,
    **_,  # ignore any other parameters
) -> tuple[bool, TextChunksWithEmbedding]:
    existing_chunks = load_chunks(
        cik=cik,
        accession_number=accession_number,
        model=embedding_model,
        dimension=dimension,
    )
    if existing_chunks and not refresh:
        logger.info(f"re-use existing chunks for {cik}/{accession_number}")
        return False, existing_chunks
    else:
        filing = SECFiling(cik=cik, accession_number=accession_number)
        text_chunks = chunk_filing(filing)
        metadata = {
            "cik": filing.cik,
            "accession_number": filing.accession_number,
            "date_filed": filing.date_filed,
            "model": embedding_model,
            "dimension": dimension,
        }
        new_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
        new_chunks.get_embeddings()
        save_chunks(new_chunks)

        logger.info(
            f"created new chunks for {cik}/{accession_number}/{embedding_model}/{dimension}"  # noqa: E501
        )
        return True, new_chunks


def extract_filing(
    cik: str,
    accession_number: str,
    embedding_model: str,
    model: str,
    dimension: int,
    **_,  # ignore any other parameters
) -> TrusteeComp | None:
    chunks = load_chunks(
        cik=cik,
        accession_number=accession_number,
        model=embedding_model,
        dimension=dimension,
    )
    queries = load_chunks(
        cik=cik,
        accession_number=accession_number,
        model=embedding_model,
        dimension=dimension,
    )

    if chunks is None or queries is None:
        return None

    result = extract_trustee_comp(queries, chunks, model)
    return result


def main(argv):
    parts = argv[1].split("/")
    data = {
        "cik": parts[1],
        "accession_number": parts[2],
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "model": DEFAULT_LLM_MODEL,
        "dimension": 768,
    }

    if parts[0] == "chunk":
        data["action"] = "chunk_one_filing"
        publish_request(data)
    elif parts[0] == "extract":
        data["action"] = "extract_one_filing"
        publish_request(data)
    else:
        print(f"Unknown command {sys.argv[1]}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    if len(sys.argv) > 1:
        main(sys.argv)
