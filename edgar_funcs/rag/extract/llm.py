import json
import logging
from datetime import datetime
from typing import Optional, Type

import litellm
from litellm import (
    Timeout,
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
)
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# from google.api_core.exceptions import (
#     DeadlineExceeded,
#     InternalServerError,
#     ResourceExhausted,
#     ServiceUnavailable,
#     TooManyRequests,
# )
# from openai import APIConnectionError, APITimeoutError, RateLimitError
# import jsonref # No longer needed
# from vertexai.generative_models import GenerativeModel # No longer needed
# from ..helper import init_vertaxai, openai_client # No longer needed


logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(
        (
            Timeout,
            APIConnectionError,
            RateLimitError,
            ServiceUnavailableError,
            InternalServerError,
        )
    ),
)
def ask_model(
    model: str, prompt: str, responseModelClass: Type[BaseModel]
) -> Optional[str]:
    try:
        start_t = datetime.now()

        messages = [{"role": "user", "content": prompt}]

        tool_name = "json_output"  # Or derive from responseModelClass.__name__
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Extract information in JSON format corresponding to {responseModelClass.__name__}.",
                    "parameters": responseModelClass.model_json_schema(),
                },
            }
        ]
        # Force the model to use the specified tool
        tool_choice = {"type": "function", "function": {"name": tool_name}}

        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0,
            tools=tools,
            tool_choice=tool_choice, # Force the model to use the defined tool
            # max_tokens, top_p etc. can be added if needed
        )

        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds():.2f} seconds"  # noqa E501
        )

        if (
            response.choices
            and response.choices[0].message.tool_calls
            and response.choices[0].message.tool_calls[0].function.arguments
        ):
            response_str = response.choices[0].message.tool_calls[0].function.arguments
            try:
                # Validate and parse with Pydantic model
                parsed_obj = responseModelClass.model_validate_json(response_str)
                return parsed_obj.model_dump_json()
            except Exception as e:
                logger.error(
                    f"Failed to parse or validate LLM JSON response for model {model} using tool calling: {e}"
                )
                logger.error(f"LLM response string (from tool arguments): {response_str}")
                return None
        else:
            logger.warning(
                f"No tool_calls or arguments in response from {model}. Response: {response}"
            )
            return None

    except RetryError as e:  # This will catch tenacity's RetryError
        logger.warning(
            f"Failed to get response from {model} after retries: {e.last_attempt.exception()}"  # noqa: E501
        )
        return None
    except litellm.exceptions.LiteLLMException as e: # Catch specific LiteLLM errors
        logger.error(f"LiteLLM API call failed for model {model}: {type(e).__name__} - {str(e)}")
        # Potentially re-raise or handle specific LiteLLM exceptions for retry
        raise # Re-raise to be caught by tenacity if configured for these exceptions
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error calling LLM model {model}: {type(e).__name__} - {str(e)}")
        return None
