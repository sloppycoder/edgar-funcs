import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict

from google.cloud import storage

from edgar import SECFiling
from rag.algo import (
    gather_chunk_distances,
    relevance_by_appearance,
    relevance_by_distance,
    relevant_chunks_with_distances,
    top_adjacent_chunks,
)
from rag.extraction import ask_model, remove_md_json_wrapper

from .chunking import ALORITHM_VERSION, chunk_text, trim_html_content
from .embedding import GEMINI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL, batch_embedding

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "gemini-1.5-flash-002"

TRUSTEE_COMP_QUERIES = [
    "Trustee Compensation Structure and Amount",
    "Independent Director or Trustee Compensation Table",
    "Board Director or Intereed Person Compensation Details with Amount",
    "Interested Person Compensation Remuneration Detailed Amount",
]


TRUSTEE_COMP_PROMPT = """
You are tasked with extracting compensation information for Trustees from a snippet
of an SEC filing 485BPOS. Here is the snippet you need to analyze:

<sec_filing_snippet>
{SEC_FILING_SNIPPET}
</sec_filing_snippet>

Your task is to extract the following information:
1. Determine if compensation information for Trustees is present in the snippet.
2. If present, extract the compensation details for each Trustee, including their name, job title, fund compensation, fund group compensation, and deferred compensation.
3. Note any additional types of compensation mentioned in the document.

Follow these steps to analyze the snippet:
1. Carefully read through the entire snippet.
2. Look for a table or section that contains compensation information for Trustees, Board Members, Board Directors, or Interested Persons.
3. If you find such information, extract the relevant details for each Trustee.
4. Pay attention to any footnotes or additional explanations related to the compensation.

Structure your output as follows:
1. A boolean field indicating whether compensation information is present in the snippet.
2. A list of Trustees with their compensation details.
3. A notes field for any additional information or explanations.

If the compensation information is not present in the snippet:
1. Set the boolean field to false.
2. Leave the list of Trustees empty.
3. In the notes field, explain that the compensation information was not found in the given snippet.

If you find any additional relevant information or need to provide explanations about your analysis,
include them in the notes field.

Provide your output in JSON format, as showsn in example below

{
 "compensation_info_present": true/false,
 "trustees": [
  {
   "year": "Year of Compensation",
   "name": "name of the trustee or N/A",
   "job_title": "the job title of the person who is a trustee. e.g. Commitee Chairperson",
   "fund_compensation": "Amount or N/A",
   "fund_group_compensation": "Amount or N/A",
   "deferred_compensation": "Amount or N/A",
   "other_compensation": {
    "type": "Amount"
   }
  }
 ],
 "notes": "Any additional information or explanations"
}
Please remove the leading $ sign and comma from compensation Amount.
"""  # noqa: E501


gcs_client = storage.Client()


class TrusteeComp(TypedDict):
    cik: str
    accession_number: str
    date_filed: str
    selected_chunks: list[int]
    selected_text: str
    response: Optional[str]
    n_trustee: int
    comp_info: dict[str, Any]


class TextChunksWithEmbedding:
    model: str
    embbedding_dimensions: int = 0
    texts: list[str] = []
    embeddings: list[list[float]] = []
    metadata: dict[str, str | int] = {}

    def __init__(self, texts: list[str], metadata: dict[str, Any] = {}):
        if not texts:
            raise ValueError("texts cannot be empty")

        self.texts = texts
        self.medata = metadata

    def is_ready(self) -> bool:
        return (
            len(self.texts) > 0
            and len(self.embeddings) > 0
            and len(self.texts) == len(self.embeddings)
            and len(self.embeddings[0]) == self.embbedding_dimensions
        )

    def get_embeddings(
        self,
        model: str = GEMINI_EMBEDDING_MODEL,
        dimension: int = 768,
    ):
        if model == OPENAI_EMBEDDING_MODEL:
            dimension = 1536

        start_t = datetime.now()
        self.embeddings = batch_embedding(self.texts, model=model, dimension=dimension)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"batch_embedding of {len(self.texts)} chunks of text with {model} took {elapsed_t.total_seconds()} seconds"  # noqa E501
        )
        self.embbedding_dimensions = dimension
        self.model = model

    def get_text_chunks(self, chunks: list[int]) -> str:
        return "\n\n".join([self.texts[i] for i in chunks])


def save_chunks(chunks: TextChunksWithEmbedding) -> None:
    if chunks.is_ready():
        path = _blob_path(chunks)
        bucket_name, prefix = _storage_prefix()
        if bucket_name:
            # means it's GCS bucket
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(prefix + path)
            blob.upload_from_string(pickle.dumps(chunks))
        else:
            # save to local file system
            output_path = Path(prefix) / path
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(chunks, f)
    else:
        raise ValueError("cannot save without embedding data")


def load_chunks(
    cik: str,
    accession_number: str,
    model: str,
    dimension: int,
    chunk_version: str = ALORITHM_VERSION,
) -> TextChunksWithEmbedding:
    path = _blob_path(
        cik=cik,
        accession_number=accession_number,
        model=model,
        dimension=dimension,
        chunk_version=chunk_version,
    )
    bucket_name, prefix = _storage_prefix()
    if bucket_name:
        # means it's GCS bucket
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(prefix + path)
        chunks = pickle.loads(blob.download_as_bytes())
    else:
        # load from local file system
        output_path = Path(prefix) / path
        with open(output_path, "rb") as f:
            chunks = pickle.load(f)
    return chunks


def chunk_filing(filing: SECFiling, method: str = "spacy"):
    if method != "spacy":
        raise ValueError(f"Unsupported chunking method {method}")

    path, content = filing.get_doc_content("485BPOS", max_items=1)[0]
    if not path.endswith((".html", ".htm")):
        raise ValueError(f"Unsupported document format {path}")

    start_t = datetime.now()
    trimmed_html = trim_html_content(content)
    chunks = chunk_text(trimmed_html, method=method)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"chunking {filing.cik}/{filing.accession_number} into {len(chunks)} chunks took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )
    return chunks


def extract_trustee_comp(
    queries: TextChunksWithEmbedding,
    chunks: TextChunksWithEmbedding,
    model=DEFAULT_LLM_MODEL,
    dimension: int = 768,
    chunk_version: str = ALORITHM_VERSION,
) -> TrusteeComp:
    if not chunks.is_ready():
        raise ValueError("embedding data is not ready")

    result: TrusteeComp = {
        "cik": chunks.medata.get("cik", ""),
        "accession_number": chunks.medata.get("accession_number", ""),
        "date_filed": chunks.medata.get("date_filed", "1971-01-01"),
        "selected_chunks": [],
        "selected_text": "",
        "n_trustee": 0,
        "response": None,
        "comp_info": {},
    }

    for method in ["distance", "appearance"]:
        relevant_chunks, relevant_text = _find_relevant_text(
            queries,
            chunks,
            method=method,
        )

        if not relevant_text or len(relevant_text) < 100:
            continue

        result["selected_chunks"] = relevant_chunks
        result["selected_text"] = relevant_text

        # step 4: send the relevant text to the LLM model with designed prompt
        response = _ask_model_about_trustee_comp(model, relevant_text)
        if response:
            try:
                result["response"] = response
                comp_info = json.loads(response)
                n_trustee = len(comp_info["trustees"])
                if n_trustee > 1:
                    result["n_trustee"] = n_trustee
                    result["comp_info"] = comp_info
                    return result

            except json.JSONDecodeError:
                pass

    return result


def _find_relevant_text(
    queries: TextChunksWithEmbedding, chunks: TextChunksWithEmbedding, method: str
):
    relevance_result = relevant_chunks_with_distances(
        queries.embeddings, chunks.embeddings, limit=20
    )
    if not relevance_result:
        return [], ""

    chunk_distances = gather_chunk_distances(relevance_result)
    if method == "distance":
        relevance_scores = relevance_by_distance(chunk_distances)
        selected_chunks = [int(s) for s in top_adjacent_chunks(relevance_scores)]
    elif method == "appearance":
        relevance_scores = relevance_by_appearance(chunk_distances)
        selected_chunks = [int(s) for s in top_adjacent_chunks(relevance_scores)]
    else:
        raise ValueError(f"Unknown method: {method}")

    return selected_chunks, chunks.get_text_chunks(selected_chunks)


def _ask_model_about_trustee_comp(model: str, relevant_text: str) -> str | None:
    start_t = datetime.now()
    prompt = TRUSTEE_COMP_PROMPT.replace("{SEC_FILING_SNIPPET}", relevant_text)
    response = ask_model(model, prompt)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )

    if response:
        return remove_md_json_wrapper(response)

    return None


def _blob_path(
    chunks: TextChunksWithEmbedding | None = None,
    cik: str = "",
    accession_number: str = "",
    chunk_version: str = "",
    model: str = "",
    dimension: int = 768,
):
    # return the path for storing a TextChunkWithEmbedding object
    if chunks and chunks.is_ready():
        # chunks object override rest of the parameters
        meta = chunks.medata
        cik = meta.get("cik", "0")
        accession_number = meta.get("accession_number", "0")
        chunk_version = meta.get("chunk_version", ALORITHM_VERSION)
        model = meta.get("model") or GEMINI_EMBEDDING_MODEL
        dimension = meta.get("dimension", 768)
    elif cik and accession_number and model and dimension:
        chunk_version = chunk_version or ALORITHM_VERSION
    else:
        raise ValueError("Must specifcy cik, accession_number, model, dimension")

    return f"{chunk_version}/{model}_{dimension}/{cik}/{accession_number}.pickle"


def _storage_prefix(storage_base_path=os.environ.get("STORAGE_PREFIX", "")):
    # return tuple of (bucket_name, prefix)
    # if the env var does not beginw with "gs://", returns ""
    if storage_base_path.startswith("gs://"):
        parts = storage_base_path[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    else:
        if storage_base_path.startswith("/"):
            return None, storage_base_path
        else:
            local_path = Path(__file__).parent.parent / storage_base_path
            return None, str(local_path)
