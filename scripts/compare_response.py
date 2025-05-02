import json

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()


"""
this script is used to compare the extraction results of two different batches
before and after switching to use structured json output

it is not part of the main pipeline, but is used to verify the correctness of the new
extraction method
"""


def do_compare():
    n_total, n_mismatch = 0, 0

    client = bigquery.Client()
    query = """
        SELECT
        cik, accession_number, extraction_type,
        new_response, old_response,
        new_chunks, old_chunks
        FROM `edgar.result_compare3`
    """
    query_job = client.query(query)

    for row in query_job:
        cik = row["cik"]
        accession_number = row["accession_number"]
        extraction_type = row["extraction_type"]
        new_response = row["new_response"]
        old_response = row["old_response"]

        new_chunks = row["new_chunks"]
        new_chunks.sort()
        new_chunks = ",".join([str(i) for i in new_chunks])
        old_chunks = row["old_chunks"]
        old_chunks.sort()
        old_chunks = ",".join([str(i) for i in old_chunks])

        n_total += 1

        try:
            response_old = json.loads(old_response)
        except json.JSONDecodeError:
            response_old = {"managers": [], "trustees": []}

        try:
            response_new = json.loads(new_response)
        except json.JSONDecodeError:
            response_new = {"managers": [], "trustees": []}

        if extraction_type == "trustee":
            n_old = len(response_old["trustees"])
            n_new = len(response_new["trustees"])
            if n_old != n_new:
                n_mismatch += 1
                print(
                    f"trustees for {cik}/{accession_number}: {n_old}/{n_new}, {old_chunks}/{new_chunks}"  # noqa E501
                )
                continue
        elif extraction_type == "fundmgr":
            n_old = len(response_old["managers"])
            n_new = len(response_new["managers"])
            if n_old != n_new:
                n_mismatch += 1
                print(
                    f"managers for {cik}/{accession_number}: {n_old}/{n_new}, {old_chunks}/{new_chunks}"  # noqa E501
                )
                continue
        else:
            print(f"?? unknown extraction type: {extraction_type}")
            continue

    print(f"total {n_total} filings compared, {n_mismatch} mismatches found")


if __name__ == "__main__":
    do_compare()

"""
SQL for creating the result_compare table:


create table `edgar.result_compare3`
as
 SELECT t1.cik, t1.accession_number, 'fundmgr' extraction_type,
 t1.response new_response, t2.response old_response,
 t1.selected_chunks new_chunks, t2.selected_chunks old_chunks
FROM `edgar.extraction_result` t1, `edgar.extraction_result` t2
WHERE t1.cik = t2.cik and t1.accession_number = t2.accession_number
and t1.batch_id = '20250503010430-nlu' and t2.batch_id = '20250422002102-ynv'
;


insert into `edgar.result_compare3`
 SELECT t1.cik, t1.accession_number, 'trustee' extraction_type,
 t1.response new_response, t2.response old_response,
 t1.selected_chunks new_chunks, t2.selected_chunks old_chunks
FROM `edgar.extraction_result` t1, `edgar.extraction_result` t2
WHERE t1.cik = t2.cik and t1.accession_number = t2.accession_number
and t1.batch_id = '20250503013026-bbb' and t2.batch_id = '20250422004812-oxk'
;
"""
