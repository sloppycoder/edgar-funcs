import argparse
import os
import random
import re
import string
from datetime import datetime
from functools import partial
from typing import Any, Hashable

import pandas as pd

from edgar_funcs.edgar import load_filing_catalog
from func_helpers import publish_message, send_cloud_run_request


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
            publish_message(data, os.getenv("REQUEST_TOPIC", ""))
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
        default=768,
        help="Embedding vector dimension, typically 768 for Gemini models and 1536 for OpenAI models ",  # noqa E501
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-005",
        help="Model to use for embedding (default: text-embedding-005)",
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
        parser.error(f"Invalid accession number format: {args.arg1}")

    return args


def main():
    args = parse_cli()

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
