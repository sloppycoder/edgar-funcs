import json
import logging
from datetime import datetime
from typing import Any, Optional, Type, TypedDict

from pydantic import BaseModel

from ..vectorize import TextChunksWithEmbedding
from .algo import (
    gather_chunk_distances,
    nearest_chunks,
    relevance_by_appearance,
    relevance_by_distance,
    top_adjacent_chunks,
)
from .llm import ask_model

logger = logging.getLogger(__name__)

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

Remove the leading $ sign and comma from compensation Amount.
"""  # noqa: E501


class TrusteeCompensation(BaseModel):
    year: str
    name: str
    job_title: str
    fund_compensation: str
    fund_group_compensation: str
    deferred_compensation: str
    other_compensation_type: Optional[str]
    other_compensation_amount: Optional[str]


class TrusteeCompensationResponse(BaseModel):
    compensation_info_present: bool
    trustees: list[TrusteeCompensation]
    notes: str


class TrusteeComp(TypedDict):
    cik: str
    accession_number: str
    date_filed: str
    selected_chunks: list[int]
    selected_text: str
    response: Optional[str]
    n_trustee: int
    comp_info: dict[str, Any]


def extract_trustee_comp_from_filing(
    cik: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_algo_version: str,
    model: str,
    **_,  # ignore any other parameters
) -> TrusteeComp | None:
    chunks = TextChunksWithEmbedding.load(
        cik=cik,
        accession_number=accession_number,
        model=embedding_model,
        dimension=embedding_dimension,
        chunk_algo_version=chunk_algo_version,
    )
    queries = _load_trustee_comp_queries(
        embedding_model=embedding_model, embedding_dimension=embedding_dimension
    )

    if chunks is None or queries is None:
        return None

    result = _extract_trustee_comp(
        queries=queries,
        chunks=chunks,
        model=model,
        responseModelClass=TrusteeCompensationResponse,
    )
    return result


def _load_trustee_comp_queries(embedding_model: str, embedding_dimension: int):
    cik, accession_number, chunk_algo_version = "0", "trustee_queries", "0"
    try:
        return TextChunksWithEmbedding.load(
            cik=cik,
            accession_number=accession_number,
            model=embedding_model,
            dimension=embedding_dimension,
            chunk_algo_version=chunk_algo_version,
        )
    except ValueError:
        # saved queries vectors not found
        # create new one
        logger.debug("Creating new trustee compensation queries")
        queries = TextChunksWithEmbedding(
            texts=TRUSTEE_COMP_QUERIES,
            metadata={
                "cik": cik,
                "accession_number": accession_number,
                "chunk_algo_version": chunk_algo_version,
            },
        )
        queries.get_embeddings(model=embedding_model, dimension=embedding_dimension)
        queries.save()
        return queries


def _extract_trustee_comp(
    queries: TextChunksWithEmbedding,
    chunks: TextChunksWithEmbedding,
    model: str,
    responseModelClass: Type[BaseModel],
) -> TrusteeComp | None:
    if not chunks.is_ready():
        raise ValueError("embedding data is not ready")

    for method in ["distance", "appearance"]:
        relevant_chunks, relevant_text = _find_relevant_text(
            queries,
            chunks,
            method=method,
        )

        if not relevant_text or len(relevant_text) < 100:
            continue

        # step 4: send the relevant text to the LLM model with designed prompt
        start_t = datetime.now()
        prompt = TRUSTEE_COMP_PROMPT.replace("{SEC_FILING_SNIPPET}", relevant_text)
        response = ask_model(model, prompt, responseModelClass)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds():.2f} seconds"  # noqa E501
        )
        if response:
            result: TrusteeComp = {
                "cik": chunks.metadata.get("cik", ""),
                "accession_number": chunks.metadata.get("accession_number", ""),
                "date_filed": chunks.metadata.get("date_filed", "1971-01-01"),
                "selected_chunks": relevant_chunks,
                "selected_text": relevant_text,
                "n_trustee": 0,
                "response": response,
                "comp_info": {},
            }
            try:
                comp_info = json.loads(response)
                n_trustee = len(comp_info["trustees"])
                if n_trustee > 1:
                    result["n_trustee"] = n_trustee
                    result["comp_info"] = comp_info
                return result

            except json.JSONDecodeError:
                pass

    return None


def _find_relevant_text(
    queries: TextChunksWithEmbedding,
    chunks: TextChunksWithEmbedding,
    method: str,
):
    relevance_result = nearest_chunks(queries.embeddings, chunks.embeddings, top_k=20)
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
