import argparse
import json
import os
import random
import re
import string
from datetime import datetime
from functools import partial
from typing import Any, Hashable

import pandas as pd
from google.cloud import firestore, pubsub_v1
from google.cloud.firestore_v1.base_query import FieldFilter

from edgar_funcs.edgar import load_filing_catalog
from func_helpers import (
    get_default_project_id,
    google_cloud_credentials,
    send_cloud_run_request,
)


def _create_publisher():
    credentials = google_cloud_credentials(
        scopes=["https://www.googleapis.com/auth/pubsub"]
    )
    if credentials:
        return pubsub_v1.PublisherClient(credentials=credentials)
    else:
        return pubsub_v1.PublisherClient()  # Use default credentials


def _publish_message(message: dict, topic_name: str):
    gcp_proj_id = get_default_project_id()
    if gcp_proj_id and topic_name:
        publisher = _create_publisher()
        topic_path = publisher.topic_path(gcp_proj_id, topic_name)

        content = json.dumps(message).encode("utf-8")
        future = publisher.publish(topic_path, content)
        future.result()  # Ensure the publish succeeds
    else:
        print(f"Invalid topic {topic_name} or project {gcp_proj_id}")


def _batch_id() -> str:
    tstamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=3))
    return f"{tstamp}-{suffix}"


def _request_payload(
    action: str,
    batch_id: str,
    cik: str,
    company_name: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    extraction_model: str,
):
    return {
        "batch_id": batch_id,
        "action": action,
        "cik": cik,
        "company_name": company_name,
        "accession_number": accession_number,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "model": extraction_model,
        "chunk_algo_version": "4",
    }


def print_stats(batch_id: str):
    """
    Count the number of records in a Firestore collection filtered by batch_id.
    """
    db = firestore.Client()
    collection_name = os.getenv("EXTRACTION_RESULT_COLLECTION", "extraction_result")
    query = db.collection(collection_name).where(
        filter=FieldFilter("batch_id", "==", batch_id)
    )

    total_docs = 0
    uniq_docs = set()
    uniq_cik = set()
    non_empty_cik = set()
    n_empty = 0

    for result in query.stream():
        total_docs += 1
        accession_number = result.get("accession_number")
        uniq_docs.add(accession_number)
        uniq_cik.add(result.get("cik"))
        if len(result.get("selected_chunks")) == 0:
            n_empty += 1
        else:
            non_empty_cik.add(result.get("cik"))

    n_docs = len(uniq_docs)
    doc_ratio = (n_docs - n_empty) / n_docs
    cik_ratio = len(non_empty_cik) / len(uniq_cik)
    print(
        f"batch_id={batch_id}: uniq/total/empty: {total_docs}/{n_docs}/{n_empty}, cik ratio:{cik_ratio:.2f}, doc ratio:{doc_ratio:.2f}, "  # noqa: E501
    )


def batch_request(todo_list: list[dict[Hashable, Any]], payload_func):
    batch_id = _batch_id()
    n_total, n_processed = len(todo_list), 0

    output_file = f"tmp/{batch_id}.csv"
    with open(output_file, "w") as f:
        f.write("batch_id,cik,company_name,accession_number\n")
        for row in todo_list:
            cik = str(row["cik"])
            accession_number = str(row["accession_number"])
            company_name = row["company_name"].strip()  # pyright: ignore
            f.write(f"{batch_id},{cik},{company_name},{accession_number}\n")
            data = payload_func(
                batch_id=batch_id,
                cik=cik,
                company_name=company_name,
                accession_number=accession_number,
            )
            _publish_message(data, os.getenv("REQUEST_TOPIC", ""))
            print(f"filing={cik}/{accession_number}")

            n_processed += 1

    print(
        f"Requested {n_processed} filings out of {n_total}, list saved to {output_file}"
    )


def parse_cli():
    parser = argparse.ArgumentParser(description="CLI for EDGAR functions")
    parser.add_argument(
        "command",
        type=str,
        choices=[
            "chunk",
            "trustee",
            "fundmgr",
            "stats",
        ],
        help="Command to execute: chunk, trustee or fundmgr",
    )
    parser.add_argument(
        "arg1",
        type=str,
        help="Accession Number to process or percentage of filings to sample",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=1536,
        help="Embedding vector dimension, typically 768 for Gemini models and 1536 for OpenAI models ",  # noqa E501
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Model to use for embedding (default: text-embedding-3-small from OpenAI)",
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default="gemini-2.0-flash",
        help="Model to use for extraction (default: gemini-2.0-flash)",
    )
    args = parser.parse_args()

    if re.match(r"^\d{10}-\d{2}-\d{6}$", args.arg1):
        args.mode = "accession_number"
    elif args.arg1.isdigit():
        args.mode = "sample"
    elif args.arg1.endswith(".csv"):
        args.mode = "list"
    else:
        if args.command == "stats":
            args.mode = "stats"
        else:
            parser.error("Invalid accession number {args.arg1}")

    return args


def main():
    args = parse_cli()

    if args.command == "stats":
        print_stats(args.arg1)
        return

    payload_func = partial(
        _request_payload,
        action=args.command,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        extraction_model=args.extraction_model,
    )

    df_filings = load_filing_catalog(args.start, args.end)
    if args.mode == "list":
        df_todo = pd.read_csv(args.arg1)
    elif args.mode == "sample":
        df_todo = df_filings.sample(frac=float(args.arg1) / 100)
        if len(df_todo) == 0:
            print("No filings to process")
            return
    elif args.mode == "accession_number":
        df_todo = df_filings[df_filings["accession_number"] == args.arg1]
        if df_todo.empty:
            print(f"Accession number {args.arg1} not found in the catalog.")
            return
    else:
        print("No accession number or percentage provided")
        return

    todo_list: [str, Any] = df_todo[["cik", "company_name", "accession_number"]].to_dict(  # pyright: ignore
        orient="records"
    )
    if len(todo_list) > 1:
        batch_request(
            todo_list=todo_list,
            payload_func=payload_func,
        )
    else:
        url = os.getenv("EDGAR_PROCESSOR_URL", "")
        todo_list[0]["batch_id"] = "single"
        result = send_cloud_run_request(url, payload_func(**todo_list[0]))
        print(f"response->\n{result['response']}")


if __name__ == "__main__":
    main()
