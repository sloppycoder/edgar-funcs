import base64
import json
import logging
import os
import sys
import traceback

import functions_framework
from cloudevents.http import CloudEvent

from func_helpers import (
    publish_message,
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
def req_processor(cloud_event: CloudEvent) -> None:
    logger.info(f"req_processor received {cloud_event.data}")
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    try:
        action = data.get("action")

        if action == "chunk_one_filing":
            is_reuse, chunks = chunk_filing_and_save_embedding(**data)
            if is_reuse:
                logger.info(f"re-use existing {len(chunks.texts)} chunks for {data}")
            else:
                logger.info(f"created new {len(chunks.texts)} chunks for {data}")

            publish_response(data, True, "success")

        elif action == "extract_one_filing":
            result = extract_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {result['n_trustee']} trustees"
                )

                result_topic = os.environ.get("TRUSTEE_RESULT_TOPIC")
                if result_topic:
                    publish_message(result, result_topic)  # type: ignore
                    message = f"result published to {result_topic}"
                else:
                    message = "result discarded"

                logger.info(message)
                publish_response(data, True, message)

        else:
            logger.info(f"Unknown action {action}")

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.info(f"chunking with {data} failed with {error_msg}\n{tb}")

        publish_response(data, False, error_msg)


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
        "embedding_dimension": 768,
        "model": DEFAULT_LLM_MODEL,
        "run_extract": True,
    }

    if parts[0] == "chunk":
        data["action"] = "chunk_one_filing"
        publish_request(data)
    elif parts[0] == "extract":
        data["action"] = "extract_one_filing"
        publish_request(data)
    elif parts[0] == "trustee":
        send_test_trustee_comp_result()
    else:
        print(f"Unknown command {sys.argv[1]}")


def send_test_trustee_comp_result():
    data = {
        "cik": "1",
        "accession_number": "1",
        "date_filed": "2022-12-01",
        "selected_chunks": [123, 456],
        "selected_text": "some_text",
        "response": "some_response",
        "n_trustee": 10,
        "comp_info": {
            "compensation_info_present": True,
            "trustees": [
                {
                    "year": "2022",
                    "name": "stuff_name_1",
                    "job_title": "title",
                    "compensation": "1000",
                },
                {
                    "year": "2022",
                    "name": "stuff_name_2",
                    "job_title": "title_stuff",
                    "compensation": "100",
                },
            ],
            "notes": "some_notes",
        },
    }
    publish_message(data, "edgarai-trustee-result")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) > 1:
        main(sys.argv)
