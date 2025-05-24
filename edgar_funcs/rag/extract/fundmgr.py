import json
import logging
from datetime import datetime
from typing import Any, Optional, TypedDict

from pydantic import BaseModel

from ..vectorize import TextChunksWithEmbedding
from .algo import (
    filter_chunks_with_keywords,
    gather_chunk_distances,
    nearest_chunks,
    relevance_by_appearance,
    relevance_by_distance,
    top_adjacent_chunks,
    top_chunks,
)
from .llm import ask_model

logger = logging.getLogger(__name__)

FUNDMGR_OWNERSHIP_KEYWORDS = [
    "portfolio manager",
    "ownership",
    "beneficial ownership",
    "dollar range",
    "securities",
    "$10,001",
    "$50,000",
    "$100,000",
    "$500,000",
    "$1,000,000",  # Common dollar values
    "10,001",
    "50,000",
    "100,000",
    "500,000",
    "1,000,000",  # Same without $ sign
    "over $1,000,000",
    "None",
    "none",
    "no ownership",  # Common ownership phrases
]


FUNDMGR_OWNERSHIP_QUERIES = [
    "Show me the beneficial ownership of each portfolio manager, including names and dollar ranges of securities they hold.",  # noqa E501
]


FUND_MGR_OWNERSHIP_PROMPT = """
SEC regulation mandates that mutual funds must disclore the securities beneciarly owned by
portofolio managers should be disclosed in filing 485BPOS. Your task is to extract
portfolio manager ownership information from a snippet from such filing.


Follow these steps to analyze the snippet:
1. Carefully read through the entire snippet.
2. Look for a table or paragraph that contains portofolio manager owernsip information.
3. If such information is found you find such information, extract the portoflio manager's name, fund name (if specified) and the manager's owenrship amount in dollar range as required by SEC. e.g. $1-$10,000, $10,001-$500,000, $500,001-$1,000,000 and over $1,000,000
4. Pay attention to any footnotes or additional explanations related ownership information, e.g. the year of disclore.

Structure your output as follows:
1. A list of managerss with their ownership details.
2. A notes field for any additional information or explanations.

If the compensation information is not present in the snippet:
1. Leave the list of managers empty.
2. In the notes field, explain that the ownership information was not found in the given snippet.

Please remove the leading $ sign from dollar range extracted

Below is a snippet you need to analyze.

<sec_filing_snippet>
{SEC_FILING_SNIPPET}
</sec_filing_snippet>

"""  # noqa: E501


class ManagerOwnership(BaseModel):
    name: str
    fund: str
    ownership_range: str


class ManagerOwnershipResponse(BaseModel):
    ownership_info_present: bool
    managers: list[ManagerOwnership]
    notes: str


class FundManagerOwnership(TypedDict):
    cik: str
    accession_number: str
    date_filed: str
    selected_chunks: list[int]
    selected_text: str
    response: Optional[str]
    ownership_info: dict[str, Any]


def extract_fundmgr_ownership_from_filing(
    cik: str,
    accession_number: str,
    embedding_model: str,
    embedding_dimension: int,  # ignored for openai embedding model
    model: str,
    chunk_algo_version: str,
    **_,  # ignore any other parameters
) -> FundManagerOwnership | None:
    chunks = TextChunksWithEmbedding.load(
        cik=cik,
        accession_number=accession_number,
        model=embedding_model,
        dimension=embedding_dimension,
        chunk_algo_version=chunk_algo_version,
    )
    queries = _load_fundmgr_ownership_queries(
        embedding_model=embedding_model, embedding_dimension=embedding_dimension
    )

    if chunks is None or queries is None:
        return None

    result = _extract_fundmgr_ownership(queries, chunks, model)
    return result


def _load_fundmgr_ownership_queries(embedding_model: str, embedding_dimension: int):
    cik, accession_number, chunk_algo_version = "0", "fundmgr_ownership_queries", "0"
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
            texts=FUNDMGR_OWNERSHIP_QUERIES,
            metadata={
                "cik": cik,
                "accession_number": accession_number,
                "chunk_algo_version": chunk_algo_version,
            },
        )
        queries.get_embeddings(model=embedding_model, dimension=embedding_dimension)
        queries.save()
        return queries


def _extract_fundmgr_ownership(
    queries: TextChunksWithEmbedding,
    chunks: TextChunksWithEmbedding,
    model: str,
) -> FundManagerOwnership | None:
    if not chunks.is_ready():
        raise ValueError("embedding data is not ready")

    for method in ["distance", "top5"]:
        relevant_chunks, relevant_text = _find_relevant_text(
            queries,
            chunks,
            method=method,
        )

        if not relevant_text or len(relevant_text) < 100:
            continue

        start_t = datetime.now()
        prompt = FUND_MGR_OWNERSHIP_PROMPT.replace("{SEC_FILING_SNIPPET}", relevant_text)
        response = ask_model(model, prompt, responseModelClass=ManagerOwnershipResponse)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds():.2f} seconds"  # noqa E501
        )
        if response:
            try:
                ownership_info = json.loads(response)
                if "managers" in ownership_info and len(ownership_info["managers"]) > 0:
                    result: FundManagerOwnership = {
                        "cik": chunks.metadata.get("cik", ""),
                        "accession_number": chunks.metadata.get("accession_number", ""),
                        "date_filed": chunks.metadata.get("date_filed", "1971-01-01"),
                        "selected_chunks": relevant_chunks,
                        "selected_text": relevant_text,
                        "response": response,
                        "ownership_info": ownership_info,
                    }
                    return result

            except json.JSONDecodeError:
                pass

    return None


def _find_relevant_text(
    queries: TextChunksWithEmbedding,
    chunks: TextChunksWithEmbedding,
    method: str,
):
    # Get filtered chunk indices using BM25
    filtered_chunk_nums = filter_chunks_with_keywords(
        chunks, keywords=FUNDMGR_OWNERSHIP_KEYWORDS, top_k=50
    )
    if not filtered_chunk_nums:
        return [], ""

    # Run embedding similarity directly with filtered indices
    relevance_result = nearest_chunks(
        queries.embeddings,
        chunks.embeddings,
        top_k=20,
        filtered_chunk_nums=filtered_chunk_nums,
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
    elif method == "top5":
        relevance_scores = relevance_by_appearance(chunk_distances)
        selected_chunks = [int(s) for s in top_chunks(relevance_scores, 5)]
    else:
        raise ValueError(f"Unknown method: {method}")

    return selected_chunks, chunks.get_text_chunks(selected_chunks)
