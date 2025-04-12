import argparse
import os
import random
import string
from datetime import datetime

import pandas as pd

from func_helpers import (
    publish_message,
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
    print(f"request_for_chunking: action={action}, batch_id={batch_id}")
    return data


def request_for_extract(
    extraction_type: str,
    batch_id,
    cik: str,
    company_name: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    extraction_model: str,
):
    action = (
        "extract_trustee_comp"
        if extraction_type == "trustee"
        else "extract_fundmgr_ownership"
    )
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
    publish_request(data)
    print(f"request_for_extract: action={action}, batch_id={batch_id}")
    return data


def send_test_extraction_result():
    extraction_result = {
        "batch_id": _batch_id(),
        "cik": "1",
        "company_name": "some_company",
        "accession_number": "1",
        "date_filed": "2022-12-01",
        "selected_chunks": [123, 456],
        "selected_text": "some_text",
        "response": "{}",
        "notes": "some_notes",
        "model": "gemini-flash-2.0",
        "extraction_type": "trustee_comp",
    }
    publish_message(extraction_result, os.environ.get("EXTRACTION_RESULT_TOPIC", ""))


def sample_catalog_and_send_requests(output_file: str, dryrun: bool, percentage: int):
    df_filings = _get_filings_by_range("2023-01-01", "2024-12-31")

    batch_id = _batch_id()
    n_total, n_processed = len(df_filings), 0

    df_sample = df_filings.sample(frac=float(percentage) / 100)

    with open(output_file, "w") as f:
        f.write("batch_id,cik,company_name,accession_number\n")
        for idx, row in df_sample.iterrows():
            cik = str(row["cik"])
            accession_number = str(row["accession_number"])
            company_name = row["company_name"].strip()  # pyright: ignore
            f.write(f"{batch_id},{cik},{company_name},{accession_number}\n")
            if not dryrun:
                request_for_chunking(
                    batch_id=batch_id,
                    cik=cik,
                    company_name=company_name,
                    accession_number=accession_number,
                    embedding_model="text-embedding-3-small",
                    embedding_dimension=1536,
                    extraction_model="gemini-2.0-flash",
                    run_extract="fundmgr",
                )

            n_processed += 1

    print(
        f"Requested {n_processed} filings out of {n_total}, list saved to {output_file}"
    )


def _batch_id() -> str:
    tstamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=3))
    return f"{tstamp}-{suffix}"


def _get_filings_by_range(start_date: str, end_date: str) -> pd.DataFrame:
    df_filings = pd.read_pickle("tests/mockdata/pickle/catalog/all_485bpos_pd.pickle")
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
            "extract",
            "publish-test-result",
            "sample",
        ],
        help="Command to execute: chunk, extract, trustee or sample",
    )
    parser.add_argument(
        "cik",
        type=str,
        nargs="?",
        default=None,
        help="CIK of the filing (optional)",
    )
    parser.add_argument(
        "accession_number",
        type=str,
        nargs="?",
        default=None,
        help="Accession Number of the filing",
    )
    parser.add_argument(
        "extract_type",
        type=str,
        choices=[
            "trustee",
            "fundmgr",
        ],
        help="Extraction type",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        default=False,
        help="Run in dry-run mode",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=1,
        help="Percentage of filings to process (default: 10)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="?",
        default=None,
        help="Input file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        default=None,
        help="Output file path",
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
    return parser.parse_args()


def main():
    args = parse_cli()

    batch_id = _batch_id()
    if args.command == "chunk":
        request_for_chunking(
            batch_id=batch_id,
            cik=args.cik,
            company_name=f"Manual trigger {args.cik}",
            accession_number=args.accession_number,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
            extraction_model=args.extraction_model,
        )
    elif args.command == "extract":
        request_for_extract(
            batch_id=batch_id,
            cik=args.cik,
            company_name=f"Manual trigger {args.cik}",
            accession_number=args.accession_number,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
            extraction_model=args.extraction_model,
            extraction_type=args.extract_type,
        )
    elif args.command == "publish-test-result":
        send_test_extraction_result()
    elif args.command == "sample":
        output_file = args.output if args.output else "tmp/sampled.csv"
        sample_catalog_and_send_requests(output_file, args.dryrun, args.percentage)
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
