import base64
import json
import logging
import os
import traceback

import functions_framework
from cloudevents.http import CloudEvent

from func_helpers import (
    publish_message,
    publish_request,
    publish_response,
    setup_cloud_logging,
)
from rag.extract.trustee import extract_trustee_comp_from_filing
from rag.vectorize import chunk_filing_and_save_embedding

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
            result = extract_trustee_comp_from_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {result['n_trustee']} trustees"
                )

                result_topic = os.environ.get("TRUSTEE_RESULT_TOPIC")
                if result_topic:
                    result_message = dict(result)
                    result_message["batch_id"] = data["batch_id"]
                    publish_message(result_message, result_topic)
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


if __name__ == "__main__":
    pass
