import logging

import tiktoken
from litellm import embedding
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def batch_embedding(
    chunks: list[str],
    model: str,
    dimension: int,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Generates embeddings for a list of text chunks using either OpenAI or
    Gemini embedding model.

    Args:
        chunks (list[str]): A list of text chunks to process

    Returns:
        list[list[float]]: A list of embeddings (one embedding per chunk)
    """
    if model.startswith("vertexai/"):
        max_tokens_per_request, max_chunks_per_request = 10000, 200
    else:
        max_tokens_per_request, max_chunks_per_request = 8191, 99999  # no limit

    # tiktoken does not support Gemini model
    # use OpenAI as stand-in.
    # since token limit is an OpenAI issue anyways.
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    embeddings = []

    # Split chunks into smaller batches based on token limit
    current_batch: list[str] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(encoding.encode(chunk))

        if chunk_tokens > max_tokens_per_request:
            # Truncate the chunk to fit within the token limit
            chunk = _truncate_chunk(chunk, encoding, max_tokens_per_request)
            chunk_tokens = len(encoding.encode(chunk))

        if (
            len(current_batch) >= max_chunks_per_request
            or (current_tokens + chunk_tokens) > max_tokens_per_request
        ):
            # Process the current batch before adding the new chunk
            result = _call_embedding_api(
                content=current_batch,
                model=model,
                task_type=task_type,
                dimension=dimension,
            )
            embeddings.extend(result)
            current_batch = []
            current_tokens = 0

        # Add the chunk to the current batch
        current_batch.append(chunk)
        current_tokens += chunk_tokens

    # Process the final batch if it exists
    if current_batch:
        result = _call_embedding_api(
            content=current_batch,
            model=model,
            task_type=task_type,
            dimension=dimension,
        )
        embeddings.extend(result)

    return embeddings


def _call_embedding_api(
    content: list[str],
    model: str,
    task_type: str,
    dimension: int,
) -> list[list[float]]:
    return _call_litellm_embedding_api(
        content,
        model=model,
        task_type=task_type,
        dimensionality=dimension,
    )


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=1, max=120),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, InternalServerError)
    ),
)
def _call_litellm_embedding_api(
    content: list[str], model: str, task_type: str, dimensionality: int
) -> list[list[float]]:
    try:
        kwargs = {}
        if model.startswith("vertexai/"):
            kwargs["task_type"] = task_type
            kwargs["dimensions"] = dimensionality

        response = embedding(model=model, input=content, **kwargs)
        return [item["embedding"] for item in response.data]
    except (RateLimitError, APIConnectionError, InternalServerError) as e:
        logger.info(f"retrying {model} embedding API call due to {type(e)}: {str(e)}")
        raise
    except Exception as e:
        logger.warning(
            f"Non-retryable error calling {model} embedding API: {type(e)}: {str(e)}"
        )
        raise


def _truncate_chunk(chunk: str, encoding, max_tokens: int) -> str:
    """
    Truncates a chunk to fit within the token limit.

    Args:
        chunk (str): The text chunk to truncate.
        encoding: The token encoding object.
        max_tokens (int): The maximum number of tokens allowed.

    Returns:
        str: The truncated text chunk.
    """
    tokens = encoding.encode(chunk)
    if len(tokens) > max_tokens:
        tokens = tokens[: max_tokens - 500]  # leave some buffer
    return encoding.decode(tokens)
