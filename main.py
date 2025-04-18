import logging
import os
import traceback

from flask import Flask, jsonify, request

from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from edgar_funcs.rag.extract.trustee import extract_trustee_comp_from_filing
from edgar_funcs.rag.vectorize import chunk_filing_and_save_embedding
from func_helpers import (
    decode_request,
    publish_message,
    setup_cloud_logging,
)

setup_cloud_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/process", methods=["POST"])
def req_processor():
    try:
        # Handle Pub/Sub push subscription message
        _, data = decode_request(request)
        if not data:
            logger.error("No Pub/Sub message received.")
            return jsonify({"error": "No Pub/Sub message received."}), 400

        action = data.get("action")
        if action not in ["chunk", "trustee", "fundmgr"]:
            logger.info(f"Unknown action {action}")
            return jsonify({"message": f"Unknown action: {action}"}), 200

        is_reuse, chunks = chunk_filing_and_save_embedding(**data)
        if is_reuse:
            logger.info(f"re-use existing {len(chunks.texts)} chunks for {data}")
        else:
            logger.info(f"created new {len(chunks.texts)} chunks for {data}")

        extraction_result = {
            "batch_id": data.get("batch_id", ""),
            "cik": data.get("cik", "0"),
            "company_name": data.get("company_name", "test company"),
            "accession_number": data.get("accession_number", "0000000000-00-000000"),
            "date_filed": data.get("date_filed", "1971-01-01"),
            "model": data["model"],
            "extraction_type": action,
        }

        if action == "trustee":
            result = extract_trustee_comp_from_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {result['n_trustee']} trustees"
                )
                extraction_result["selected_chunks"] = result["selected_chunks"]
                extraction_result["selected_text"] = result["selected_text"]
                extraction_result["response"] = result["response"]

        elif action == "fundmgr":
            result = extract_fundmgr_ownership_from_filing(**data)
            if result:
                logger.info(
                    f"extraction with {data} found {len(result['ownership_info']['managers'])} managers"  # noqa E501
                )
                extraction_result["selected_chunks"] = result["selected_chunks"]
                extraction_result["selected_text"] = result["selected_text"]
                extraction_result["response"] = result["response"]

        else:
            extraction_result["selected_chunks"] = []
            extraction_result["selected_text"] = ""
            extraction_result["response"] = ""

        _publish_result(extraction_result)

        return jsonify(extraction_result), 200

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        data = request.get_json()
        logger.error(f"chunking with {data} failed with {error_msg}\n{tb}")

        # must return 200 in order to indicate the message
        # has been processed
        return jsonify({"error": error_msg}), 200


def _publish_result(result: dict):
    result_topic = os.environ.get("EXTRACTION_RESULT_TOPIC")
    if result_topic:
        publish_message(result, result_topic)
        logger.info(f"result published to {result_topic}")
    else:
        logger.info("result discarded")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
