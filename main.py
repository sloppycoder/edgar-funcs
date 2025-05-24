import logging
import os
import traceback
from datetime import datetime
from typing import Any

from flask import Flask, jsonify, request

from edgar_funcs.edgar import SECFiling
from edgar_funcs.rag.extract.fundmgr import extract_fundmgr_ownership_from_filing
from edgar_funcs.rag.extract.trustee import extract_trustee_comp_from_filing
from edgar_funcs.rag.vectorize import (
    TextChunksWithEmbedding,
    _blob_path,
    chunk_filing,
)
from func_helpers import (
    decode_request,
    delete_lock,
    publish_message,
    setup_cloud_logging,
    write_lock,
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

        logger.info(f"Received request for {data}")

        # check if request action is valid
        action = data.get("action")
        if action not in ["chunk", "trustee", "fundmgr"]:
            logger.info(f"Unknown action {action}")
            return jsonify({"error": f"Unknown action: {action}"}), 200

        # step 1: check if text chunks and embeddings already exists
        chunks = _retrieve_chunks_for_filing(**data)
        if not chunks:
            msg = "cannot retrieve text chunks, maybe another job is in progress?"
            logger.info(msg)
            return jsonify({"error": msg}), 200

        # step 2: perform extraction using LLM
        data["date_filed"] = chunks.metadata.get("date_filed", "1971-01-01")
        result = _perform_extraction(data)

        # step 3: save the extraction result
        _publish_result(result)

        return jsonify(result), 200

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        data = request.get_json()
        logger.error(f"processing with {data} failed with {error_msg}\n{tb}")
        # must return 200 in order to indicate the message
        # has been processed
        return jsonify({"error": error_msg}), 200


def _retrieve_chunks_for_filing(
    cik: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_algo_version: str,
    batch_id: str,
    **_,  # ignore any other parameters
) -> TextChunksWithEmbedding | None:
    try:
        existing_chunks = TextChunksWithEmbedding.load(
            cik=cik,
            accession_number=accession_number,
            model=embedding_model,
            dimension=embedding_dimension,
            chunk_algo_version=chunk_algo_version,
        )
        logger.info(
            f"re-use {len(existing_chunks.texts)} chunks for {cik}/{accession_number}"
        )
        return existing_chunks
    except ValueError:
        # saved chunks not found
        pass

    metadata = {
        "cik": cik,
        "accession_number": accession_number,
        "chunk_algo_version": chunk_algo_version,
        "model": embedding_model,
        "dimension": embedding_dimension,
    }

    lock_blob_path = _blob_path(**metadata).replace(".pickle", "_lock.json")
    if not write_lock(lock_blob_path):
        return None

    start_t = datetime.now()
    filing = SECFiling(cik=cik, accession_number=accession_number)
    metadata["date_filed"] = filing.date_filed
    text_chunks = chunk_filing(filing)
    new_chunks = TextChunksWithEmbedding(text_chunks, metadata=metadata)
    new_chunks.get_embeddings(model=embedding_model, dimension=embedding_dimension)
    new_chunks.save()
    elapsed_t = datetime.now() - start_t
    logger.info(
        f"created new {len(new_chunks.texts)} chunks for {cik}/{accession_number} took {elapsed_t.total_seconds():.2f} seconds"  # noqa E501
    )
    delete_lock(lock_blob_path)
    return new_chunks


def _perform_extraction(data: dict) -> dict[str, Any]:
    action = data.get("action")
    cik = data.get("cik", "0")
    accession_number = data.get("accession_number", "0000000000-00-000000")
    extraction_result = {
        "batch_id": data.get("batch_id", ""),
        "cik": cik,
        "company_name": data.get("company_name", "test company"),
        "accession_number": accession_number,
        "date_filed": data.get("date_filed", "1971-01-01"),
        "model": data["model"],
        "extraction_type": action,
        "selected_chunks": [],
        "selected_text": "N/A",
        "response": "N/A",
    }

    # step 2: perform extraction if action is trustee or fundmgr
    if action == "trustee":
        result = extract_trustee_comp_from_filing(**data)
        if result:
            logger.info(f"extraction with {data} found {result['n_trustee']} trustees")
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
        # action is "chunk", just load the chunks to verify it exists
        chunks = TextChunksWithEmbedding.load(
            cik=cik,
            accession_number=accession_number,
            model=data.get("embedding_model", "_dummy_"),
            dimension=data.get("embedding_dimension", "0"),
            chunk_algo_version=data.get("chunk_algo_version", "0"),
        )
        extraction_result["selected_chunks"] = []
        extraction_result["selected_text"] = ""
        extraction_result["response"] = "" if chunks else "no chunks found"

    return extraction_result


def _publish_result(result: dict):
    batch_id = result["batch_id"]
    id = f"{result['extraction_type']}/{batch_id}/{result['cik']}/{result['accession_number']}"  # noqa E501
    result_topic = os.environ.get("EXTRACTION_RESULT_TOPIC")
    if result_topic and not batch_id.startswith("single"):
        publish_message(result, result_topic)
        logger.info(f"result for {id} published to {result_topic}")
    else:
        logger.info(f"result for {id} discarded")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
