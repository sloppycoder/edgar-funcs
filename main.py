import json
import logging
import sys
import traceback

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
)
from rag.embedding import GEMINI_EMBEDDING_MODEL

setup_cloud_logging()
logger = logging.getLogger(__name__)

REQ_CHUNK = "edgar.funcs.chunk_filing.req"
RESP_CHUNK = "edgar.funcs.chunk_filing.resp"
REQ_EXTRACT_TRUSTEE = "edgar.funcs.extract_trustee_comp.req"


@functions_framework.cloud_event
def req_processor(cloud_event: CloudEvent):
    logger.info(f"req_processor received {cloud_event}")

    event_type = cloud_event["type"]
    data = cloud_event.data

    if event_type == REQ_CHUNK:
        try:
            refresh = data.get("refresh", False)

            existing_chunks = load_chunks(**data)
            if existing_chunks and not refresh:
                logger.info(f"re-use existing chunks for  {data}")
                publish_response(event_type, data, True, "re-use")
            else:
                chunk_filing_and_save_embedding(**data)
                logger.info(f"created new chunks with {data}")
                publish_response(event_type, data, True, "success")

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.info(f"chunking with {data} failed with {error_msg}\n{tb}")
            publish_response(event_type, data, True, "error_msg")

    elif event_type == REQ_EXTRACT_TRUSTEE:
        try:
            result = extract_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {result['n_trustee']} trustees"
                )
                publish_response(event_type, data, True, json.dumps(result))
            else:
                logger.info(f"extraction with {data} failed")
                publish_response(event_type, data, False, "failed")

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.info(f"extraction with {data} failed with {error_msg}\n{tb}")
            publish_response(event_type, data, True, "error_msg")
    else:
        # don't know how to process an event
        # it's probably meant for some other processors
        pass


@functions_framework.cloud_event
def resp_processor(cloud_event: CloudEvent):
    logger.info(f"resp_processor received {cloud_event}")

    event_type = cloud_event["type"]
    data = cloud_event.data

    if event_type == RESP_CHUNK:
        try:
            params = data["params"]
            if params.get("run_extract", False):
                publish_request(REQ_EXTRACT_TRUSTEE, params)
                logger.info(f"sent request for extraction with {params}")

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logger.info(f"chunking with {data} failed with {error_msg}\n{tb}")
            publish_response(event_type, data, True, "error_msg")

    else:
        # don't know how to process an event
        # it's probably meant for some other processors
        pass


def chunk_filing_and_save_embedding(
    cik: str,
    accession_number: str,
    embedding_model: str,
    dimension: int,
    **_,  # ignore any other parameters
):
    filing = SECFiling(cik=cik, accession_number=accession_number)
    text_chunks = chunk_filing(filing)

    metadata = {
        "cik": filing.cik,
        "accession_number": filing.accession_number,
        "date_filed": filing.date_filed,
        "model": embedding_model,
        "dimension": dimension,
    }
    filing_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
    filing_chunks.get_embeddings()
    return filing_chunks


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
        event_type = REQ_CHUNK
        publish_request(event_type, data)
    elif parts[0] == "extract":
        event_type = REQ_EXTRACT_TRUSTEE
        publish_request(event_type, data)
    else:
        print(f"Unknown command {sys.argv[1]}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    if len(sys.argv) > 1:
        main(sys.argv)
