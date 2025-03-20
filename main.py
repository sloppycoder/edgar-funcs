import base64
import json
import logging
import sys
import traceback
from typing import Any

import functions_framework
from cloudevents.http import CloudEvent

from func_helpers import (
    publish_request,
    publish_response,
    setup_cloud_logging,
)
from rag.extract.llm import DEFAULT_LLM_MODEL
from rag.extract.trustee import extract_filing
from rag.vectorize import chunk_filing_and_save_embedding
from rag.vectorize.embedding import GEMINI_EMBEDDING_MODEL

setup_cloud_logging()
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def req_processor(cloud_event: CloudEvent):
    logger.info(f"req_processor received {cloud_event.data}")
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    try:
        if dispatch_event(data):
            publish_response(data, True, "success")
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
    params = data["params"]
    action = params["action"]

    if action == "chunk_one_filing":
        req_is_success = data["success"]
        if req_is_success and params.get("run_extract", False):
            params["action"] = "extract_one_filing"
            publish_request(params)
            logger.info(f"publish request for {params}")


def main(argv):
    parts = argv[1].split("/")
    data = {
        "cik": parts[1],
        "accession_number": parts[2],
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "model": DEFAULT_LLM_MODEL,
        "dimension": 768,
        "run_extract": True,
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
