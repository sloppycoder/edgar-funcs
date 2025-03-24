import random
import string
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from func_helpers import (
    publish_message,
    publish_request,
)
from rag.extract.llm import DEFAULT_LLM_MODEL
from rag.vectorize.embedding import GEMINI_EMBEDDING_MODEL

load_dotenv()


def request_for_chunking(
    batch_id: str,
    cik: str,
    accession_number: str,
    run_extract: bool = False,
):
    data = {
        "batch_id": batch_id,
        "action": "chunk_one_filing",
        "cik": cik,
        "accession_number": accession_number,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "embedding_dimension": 768,
        "model": DEFAULT_LLM_MODEL,
        "run_extract": run_extract,
    }
    publish_request(data)
    return data


def request_for_extract(batch_id, cik: str, accession_number: str):
    data = {
        "batch_id": batch_id,
        "action": "extract_one_filing",
        "cik": cik,
        "accession_number": accession_number,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "embedding_dimension": 768,
        "model": DEFAULT_LLM_MODEL,
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


def sample_catalog_and_send_requests():
    batch_id = _batch_id()

    df_filings = pd.read_pickle("tests/mockdata/pickle/catalog/all_485bpos_pd.pickle")
    assert len(df_filings) > 10000

    df_cik = pd.read_csv("tests/mockdata/misc/cik.csv")
    assert len(df_cik) > 1000

    df_filtered = df_filings[
        (df_filings["date_filed"] > "2004-01-01")
        & (df_filings["date_filed"] < "2004-12-31")
    ]
    df_sample = (pd.merge(df_filtered, df_cik, on="cik")).sample(30)

    for idx, row in df_sample.iterrows():
        cik = str(row["cik"])
        accession_number = row["accession_number"]
        print(f"requesting {batch_id},{cik},{accession_number}")
        request_for_chunking(batch_id, cik, accession_number, run_extract=True)


def _batch_id():
    tstamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=3))
    return f"{tstamp}-{suffix}"


def main(args):
    if args[0] == "chunk" and len(args) >= 3:
        request_for_chunking(_batch_id(), args[1], args[2])
    elif args[0] == "extract" and len(args) >= 3:
        request_for_extract(_batch_id(), args[1], args[2])
    elif args[0] == "trustee":
        send_test_trustee_comp_result()
    elif args[0] == "sample":
        sample_catalog_and_send_requests()
    else:
        print("Unknown command")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("nothing to do")
