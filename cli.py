import argparse
import gzip
import json
import random
import string
from datetime import datetime

import pandas as pd
from google.cloud import bigquery

from func_helpers import (
    model_settings,
    publish_message,
    publish_request,
)


def request_for_chunking(
    batch_id: str,
    cik: str,
    accession_number: str,
    run_extract: bool = False,
):
    embedding_model, embedding_dimension, extraction_model = model_settings()
    data = {
        "batch_id": batch_id,
        "action": "chunk_one_filing",
        "cik": cik,
        "accession_number": accession_number,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "model": extraction_model,
        "run_extract": run_extract,
    }
    publish_request(data)
    return data


def request_for_extract(batch_id, cik: str, accession_number: str):
    embedding_model, embedding_dimension, extraction_model = model_settings()
    data = {
        "batch_id": batch_id,
        "action": "extract_one_filing",
        "cik": cik,
        "accession_number": accession_number,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "model": extraction_model,
    }
    publish_request(data)
    return data


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


def sample_catalog_and_send_requests(output_file: str, dryrun: bool, percentage: int):
    df_filings = _get_filings_by_range("2004-01-01", "2004-12-31")

    batch_id = _batch_id()
    n_total, n_processed = len(df_filings), 0

    df_sample = df_filings.sample(frac=float(percentage) / 100)

    with gzip.open(output_file, "wt") as f:
        f.write("batch_id,cik,accession_number\n")
        for idx, row in df_sample.iterrows():
            cik = str(row["cik"])
            accession_number = row["accession_number"]
            f.write(f"{batch_id},{cik},{accession_number}\n")
            if not dryrun:
                request_for_chunking(batch_id, cik, accession_number, run_extract=True)

            n_processed += 1

    print(
        f"Requested {n_processed} filings out of {n_total}, list saved to {output_file}"
    )


def export_process_result(input_file: str, output_file: str):
    df_processed = pd.read_csv(input_file)
    df_processed["cik"] = df_processed["cik"].astype(str)
    assert len(df_processed) > 0, "No processed data found in the input file"

    batch_id = df_processed["batch_id"].iloc[0]
    df_result = pd.DataFrame(_query_result(batch_id))

    # Perform a left join with df_procesed on the left side
    df_merged = pd.merge(
        df_processed, df_result, on=["cik", "accession_number"], how="left"
    )

    df_filings = _get_filings_by_range("2000-01-01", "2024-12-31")
    # Add company_name column by looking up from df_filings
    df_merged = pd.merge(
        df_merged,
        df_filings[["cik", "accession_number", "company_name"]],
        on=["cik", "accession_number"],
        how="left",
    )

    # Ensure num_trustees column is set to 0 if NaN
    df_merged["num_trustees"] = df_merged["num_trustees"].fillna(0)

    # Write each row as a JSON object to the output file in JSONL format
    n_count = 0
    with gzip.open(output_file, "wt") as f:
        for _, row in df_merged.iterrows():
            row_dict = row.to_dict()
            n_count += 1
            f.write(json.dumps(row_dict) + "\n")
    print(f"Exported {n_count} records to {output_file}")


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
    df_result = pd.merge(df_filtered, df_cik, on="cik")
    df_result["cik"] = df_result["cik"].astype(str)
    return df_result


def _query_result(batch_id: str) -> list[dict]:
    """
    Query BigQuery table edgar2.trustee_comp_result to
    get the results for a specific batch_id.
    """
    client = bigquery.Client()
    query = """
        SELECT
            cik, accession_number, date_filed,
            selected_chunks as chunks_used,
            selected_text as relevant_text,
            n_trustee as num_trustees,
            response trustees_comp
        FROM `edgar2.trustee_comp_result`
        WHERE batch_id = @batch_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("batch_id", "STRING", batch_id),
        ]
    )
    query_job = client.query(query, job_config=job_config)
    return [dict(row) for row in query_job.result()]


def parse_cli():
    parser = argparse.ArgumentParser(description="CLI for EDGAR functions")
    parser.add_argument(
        "command",
        type=str,
        choices=["chunk", "extract", "test", "sample", "export"],
        help="Command to execute: chunk, extract, trustee, sample, or export",
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
        "--dryrun",
        action="store_true",
        default=False,
        help="Run in dry-run mode",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=20,
        help="Percentage of filings to process (default: 20)",
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
    return parser.parse_args()


def main():
    args = parse_cli()

    if args.command == "chunk":
        request_for_chunking(_batch_id(), args.cik, args.accession_number)
    elif args.command == "extract":
        request_for_extract(_batch_id(), args.cik, args.accession_number)
    elif args.command == "test":
        send_test_trustee_comp_result()
    elif args.command == "sample":
        output_file = args.output if args.output else "tmp/processed.csv.gz"
        sample_catalog_and_send_requests(output_file, args.dryrun, args.percentage)
    elif args.command == "export":
        input_file = args.input if args.input else "tmp/processed.csv"
        output_file = args.output if args.output else "tmp/seed_data.jsonl.gz"
        export_process_result(input_file, output_file)
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
