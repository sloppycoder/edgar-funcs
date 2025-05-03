import json

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

_columns_to_select = (
    " batch_id, cik, accession_number, extraction_type, response, selected_chunks, model"
)
_old_batch_id = "20250422004812-oxk"
_new_batch_id = "20250503013026-bbb"
# _old_batch_id = "20250422002102-ynv"
# _new_batch_id = "20250503010430-nlu"


def _bq_to_df(query):
    client = bigquery.Client()
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return df


def _format_json(json_str):
    try:
        if json_str:
            obj = json.loads(json_str)
            return json.dumps(obj, indent=2)
    except json.JSONDecodeError:
        return json_str


def _chunks_as_str(chunks_array):
    if chunks_array.size > 0:
        return ", ".join([str(chunk) for chunk in sorted(chunks_array)])
    return ""


def _n_entity(json_str, key):
    try:
        if json_str:
            obj = json.loads(json_str)
            if key in obj:
                return len(obj[key])
    except json.JSONDecodeError:
        pass

    return 0


def prep_compare_df(old_batch_id, new_batch_id):
    _df_new = _bq_to_df(
        f"""
        SELECT {_columns_to_select}
        FROM `edgar.extraction_result`
        WHERE batch_id = '{new_batch_id}'
        """
    )
    _df_old = _bq_to_df(
        f"""
        SELECT {_columns_to_select}
        FROM `edgar.extraction_result` t
        WHERE batch_id = '{old_batch_id}'
        AND EXISTS (
            SELECT 1
            FROM `edgar.extraction_result`
            WHERE batch_id = '{new_batch_id}'
            AND cik = t.cik
            AND accession_number = t.accession_number
            AND extraction_type = t.extraction_type
        )
        """
    )
    # join 2 df on cik and accession_number fields
    _df_compare = _df_new.merge(
        _df_old,
        on=["cik", "accession_number"],
        suffixes=("_new", "_old"),
        how="inner",
    )

    if _df_compare["extraction_type_new"].iloc[0] == "trustee":
        key = "trustees"
    else:
        key = "managers"

    _df_compare["n_entity_new"] = _df_compare["response_new"].apply(
        lambda x: _n_entity(x, key)
    )
    _df_compare["n_entity_old"] = _df_compare["response_old"].apply(
        lambda x: _n_entity(x, key)
    )
    # filter out rows where n_entity_new and n_entity_old are equal
    _df_compare = _df_compare[_df_compare["n_entity_new"] != _df_compare["n_entity_old"]]
    _df_compare["chunks_new"] = _df_compare["selected_chunks_new"].apply(
        lambda x: _chunks_as_str(x)
    )
    _df_compare["chunks_old"] = _df_compare["selected_chunks_old"].apply(
        lambda x: _chunks_as_str(x)
    )
    _df_compare["response_new"] = _df_compare["response_new"].apply(
        lambda x: _format_json(x)
    )
    _df_compare["response_old"] = _df_compare["response_old"].apply(
        lambda x: _format_json(x)
    )

    _df_compare = _df_compare.drop(
        columns=[
            "batch_id_new",
            "batch_id_old",
            "selected_chunks_new",
            "selected_chunks_old",
            "extraction_type_old",
        ]
    )

    return _df_compare
