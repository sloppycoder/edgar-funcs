import argparse
import random
import re
import string
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Hashable

import pandas as pd

from func_helpers import (
    publish_request,
)


def request_for_chunking(
    batch_id: str,
    cik: str,
    company_name: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    extraction_model: str,
    run_extract: str = "",
):
    action = "chunk_one_filing"
    data = {
        "batch_id": batch_id,
        "action": action,
        "cik": cik,
        "company_name": company_name,
        "accession_number": accession_number,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "model": extraction_model,
        "run_extract": run_extract,
        "chunk_algo_version": "4",
    }
    publish_request(data)
    print(
        f"chunking: {run_extract}, batch_id={batch_id}, filing={cik}/{accession_number}, "  # noqa E501
    )
    return data


def batch_request(todo_list: list[dict[Hashable, Any]], output_file: str, request_func):
    batch_id = _batch_id()
    n_total, n_processed = len(todo_list), 0

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


@lru_cache(maxsize=1)
def _get_filings_by_range(start_date: str, end_date: str) -> pd.DataFrame:
    catalog_path = Path(__file__).parent / "data/catalog/all_485bpos_pd.pickle"
    df_filings = pd.read_pickle(catalog_path)
    assert len(df_filings) > 10000

    df_cik = pd.read_csv("tests/mockdata/misc/cik.csv")
    assert len(df_cik) > 1000

    df_filtered = df_filings[
        (df_filings["date_filed"] > start_date) & (df_filings["date_filed"] < end_date)
    ]
    df_result = pd.merge(df_filtered, df_cik, on="cik")  # pyright: ignore
    df_result["cik"] = df_result["cik"].astype(str)
    return df_result


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
        default="gemini-flash-2.0",
        help="Model to use for extraction (default: gemini-flash-2.0)",
    )
    args = parser.parse_args()

    if re.match(r"^\d{10}-\d{2}-\d{6}$", args.arg1):
        args.accession_number = args.arg1
        args.percentage = 0
    elif args.arg1.isdigit():
        args.percentage = int(args.arg1)
    else:
        parser.error(f"Invalid accession number format: {args.arg1}")

    return args


def main():
    args = parse_cli()

    if args.command == "chunk":
        run_extract = ""
    elif args.command == "trustee":
        run_extract = "trustee"
    elif args.command == "fundmgr":
        run_extract = "fundmgr"
    else:
        print(f"Unknown command: {args.command}")
        return

    request_func = partial(
        request_for_chunking,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        extraction_model=args.extraction_model,
        run_extract=run_extract,
    )

    df_filings = _get_filings_by_range(args.start, args.end)
    if args.percentage:
        df_todo = df_filings.sample(frac=float(args.percentage) / 100)
        if len(df_todo) == 0:
            print("No filings to process")
            return
    elif args.accession_number:
        df_todo = df_filings[df_filings["accession_number"] == args.accession_number]
        if df_todo.empty:
            print(f"Accession number {args.accession_number} not found in the catalog.")
            return
    else:
        print("No accession number or percentage provided")
        return

    todo_list = df_todo[["cik", "company_name", "accession_number"]].to_dict(  # pyright: ignore
        orient="records"
    )
    batch_request(
        todo_list=todo_list,
        output_file="tmp/processed.csv",
        request_func=request_func,
    )


if __name__ == "__main__":
    main()
