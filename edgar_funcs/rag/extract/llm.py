import logging
from datetime import datetime
from typing import Optional, Type

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def ask_model(
    model: str, prompt: str, responseModelClass: Type[BaseModel]
) -> Optional[str]:
    """
    Ask a language model to generate a response based on a prompt with structured output.

    Args:
        model: The model name (e.g., "gpt-4o", "vertex_ai/gemini-2.0-flash-001")
        prompt: The text prompt to send to the model
        responseModelClass: Pydantic model class defining the expected response structure

    Returns:
        JSON string representation of the structured response, or None if the request fails

    Example:
        >>> class MathResponse(BaseModel):
        ...     answer: str
        ...     explanation: str
        >>> result = ask_model("gpt-4o", "What is 2+2?", MathResponse)
        >>> # Returns: '{"answer": "4", "explanation": "Adding 2 and 2 equals 4"}'
    """  # noqa: E501
    try:
        start_t = datetime.now()
        response = _chat_with_litellm(model, prompt, responseModelClass)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds():.2f} seconds"  # noqa E501
        )
        return response

    except RetryError as e:
        logger.warning(
            f"Failed to get response from {model} after retries: {e.last_attempt.exception()}"  # noqa: E501
        )
        return None


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(
        (
            RateLimitError,
            APIConnectionError,
            Timeout,
            ServiceUnavailableError,
            APIError,
        )
    ),
)
def _chat_with_litellm(
    model_name: str,
    prompt: str,
    responseModelClass: Type[BaseModel],
) -> Optional[str]:
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
            response_format=responseModelClass,
        )

        content = response.choices[0].message.content
        if content:
            # Validate the response matches the expected schema
            try:
                parsed_data = responseModelClass.model_validate_json(content)
                return parsed_data.model_dump_json()
            except Exception as e:
                logger.warning(f"Response validation failed: {e}")
                return content

    except (
        RateLimitError,
        APIConnectionError,
        Timeout,
        ServiceUnavailableError,
        APIError,
    ) as e:
        logger.info(f"retrying {model_name} API call due to {type(e)}: {str(e)}")
        raise
    except Exception as e:
        logger.warning(f"Non-retryable error calling {model_name}: {type(e)}: {str(e)}")
        return None

    return None
