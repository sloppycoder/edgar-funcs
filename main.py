import base64
import json
import logging
import os
import traceback

import functions_framework
from cloudevents.http import CloudEvent

from edgar_funcs.func_helpers import (
    publish_message,
    publish_request,
    publish_response,
    setup_cloud_logging,
)
from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from edgar_funcs.rag.extract.trustee import extract_trustee_comp_from_filing
from edgar_funcs.rag.vectorize import chunk_filing_and_save_embedding

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

        elif action == "extract_trustee_comp":
            result = extract_trustee_comp_from_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {result['n_trustee']} trustees"
                )
                pub_data = dict(result)
                pub_data["company_name"] = data["company_name"]
                _publish_result(data, pub_data)

                publish_response(data, True, "result published")
            else:
                publish_response(data, False, "no info extracted")

        elif action == "extract_fundmgr_ownership":
            result = extract_fundmgr_ownership_from_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {len(result['ownership_info']['managers'])} managers"  # noqa E501
                )
                pub_data = dict(result)
                pub_data["company_name"] = data["company_name"]
                _publish_result(data, pub_data)

                publish_response(data, True, "result published")
            else:
                publish_response(data, False, "no info extracted")

        else:
            logger.info(f"Unknown action {action}")
            return

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"chunking with {data} failed with {error_msg}\n{tb}")

        publish_response(data, False, error_msg)


@functions_framework.cloud_event
def resp_processor(cloud_event: CloudEvent):
    logger.info(f"resp_processor received {cloud_event.data}")
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    params = data["params"]
    action = params["action"]

    if action == "chunk_one_filing":
        req_is_success = data["success"]
        extract_type = params.get("run_extract")
        if req_is_success and extract_type:
            params["action"] = (
                "extract_trustee_comp"
                if "trustee" in extract_type
                else "extract_fundmgr_ownership"
            )
            publish_request(params)
            logger.info(f"publish request for {params}")


def _publish_result(data: dict, result: dict):
    if not result:
        logger.info(f"No data extracted for {data}")
        publish_response(data, False, "info not found")
        return

    extraction_result = {
        "batch_id": data.get("batch_id", ""),
        "cik": result.get("cik", ""),
        "company_name": result.get("company_name", ""),
        "accession_number": result.get("accession_number", ""),
        "date_filed": result.get("date_filed", ""),
        "selected_chunks": result.get("selected_chunks", []),
        "selected_text": result.get("selected_text", ""),
        "response": result.get("response", ""),
        "notes": result.get("notes", ""),
        "model": data["model"],
        "extraction_type": data.get("run_extract", ""),
    }

    result_topic = os.environ.get("EXTRACTION_RESULT_TOPIC")
    if result_topic:
        publish_message(extraction_result, result_topic)
        message = f"result published to {result_topic}"
    else:
        message = "result discarded"

    logger.info(message)
    publish_response(data, True, message)


if __name__ == "__main__":
    pass
