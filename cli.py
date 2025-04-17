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
from func_helpers import publish_message


def send_request(
    action: str,
    batch_id: str,
    cik: str,
    company_name: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    extraction_model: str,
):
    data = {
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
    publish_message(data, os.getenv("REQUEST_TOPIC", ""))
    print(f"{action}: filing={cik}/{accession_number}")
    return data


def batch_request(todo_list: list[dict[Hashable, Any]], request_func):
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
            request_func(
                batch_id=batch_id,
                cik=cik,
                company_name=company_name,
                accession_number=accession_number,
            )

            n_processed += 1

    print(
        f"Requested {n_processed} filings out of {n_total}, list saved to {output_file}"
    )


def _batch_id() -> str:
    tstamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=3))
    return f"{tstamp}-{suffix}"


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

    request_func = partial(
        send_request,
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

    todo_list = df_todo[["cik", "company_name", "accession_number"]].to_dict(  # pyright: ignore
        orient="records"
    )
    batch_request(
        todo_list=todo_list,
        request_func=request_func,
    )


if __name__ == "__main__":
    main()
