import json
import logging
from typing import Optional, Type

import jsonref
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
)
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.generative_models import GenerativeModel

from ..helper import init_vertaxai, openai_client

logger = logging.getLogger(__name__)


def ask_model(
    model: str, prompt: str, responseModelClass: Type[BaseModel]
) -> Optional[str]:
    try:
        if model.startswith("gemini"):
            google_schema = _convert_json_schema_to_google_schema(
                responseModelClass.model_json_schema()
            )
            return _chat_with_gemini(model, prompt, google_schema)
        elif model.startswith("gpt"):
            return _chat_with_gpt(model, prompt, responseModelClass)
        else:
            raise ValueError(f"Unknown model: {model}")
    except RetryError as e:
        logger.warning(
            f"Failed to get response from {model} after retries: {e.last_attempt.exception()}"  # noqa: E501
        )


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(
        (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
        )
    ),
)
def _chat_with_gpt(
    model_name: str,
    prompt: str,
    responseModelClass: Type[BaseModel],
) -> Optional[str]:
    client = openai_client()

    try:
        response = client.responses.parse(
            model=model_name,
            temperature=0,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=8192,
            text_format=responseModelClass,
        )

        if response.output_parsed:
            return response.output_parsed.model_dump_json()

    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        logging.info(f"retrying OpenAI API call due to {type(e)}")
        raise
    except Exception as e:
        logging.warning(f"Error calling OpenAI API: {type(e)},{str(e)}")

    return None


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(
        (
            DeadlineExceeded,  # timeout
            ServiceUnavailable,  # transient unavailability
            TooManyRequests,  # rate limit exceeded
            ResourceExhausted,  # quota or rate limit exceeded
            InternalServerError,  # 500-level errors
        )
    ),
)
def _chat_with_gemini(model_name: str, prompt: str, output_schema) -> Optional[str]:
    try:
        init_vertaxai()
        model = GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "temperature": 0,
                "top_p": 0.95,
                "response_mime_type": "application/json",
                "response_schema": output_schema,
            },
        )
        return response.text

    except (
        DeadlineExceeded,  # timeout
        ServiceUnavailable,  # transient unavailability
        ResourceExhausted,  # quota or rate limit exceeded
        TooManyRequests,  # rate limit exceeded
        InternalServerError,  # 500-level errors
    ) as e:
        logging.info(f"retrying Gemini API call due to {type(e)}")
        raise
    except Exception as e:
        logging.warning(f"Error calling Gemini API: {type(e)},{str(e)}")
        return None


def _convert_json_schema_to_google_schema(json_schema: dict) -> dict:  # noqa: C901
    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "object": "OBJECT",
        "array": "ARRAY",
        "null": "NULL",
    }

    def convert(schema, parent_required=None):
        if not isinstance(schema, dict):
            return schema

        if "anyOf" in schema:
            not_nones = [item for item in schema["anyOf"] if item.get("type") != "null"]
            return convert(not_nones[0]) if not_nones else {}

        google_schema = {}

        # Convert and normalize type
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            # Remove 'null' from type (used for Optional fields)
            clean_types = [t for t in schema_type if t != "null"]
            chosen_type = clean_types[0] if clean_types else "string"
            google_schema["type"] = type_map.get(chosen_type, chosen_type.upper())
        elif isinstance(schema_type, str):
            google_schema["type"] = type_map.get(schema_type, schema_type.upper())

        # Convert object properties
        if "properties" in schema:
            google_schema["properties"] = {}
            google_required = []

            for prop_name, prop_schema in schema["properties"].items():
                is_optional = "anyOf" in prop_schema and any(
                    "null" == item["type"] for item in prop_schema["anyOf"]
                )
                google_schema["properties"][prop_name] = convert(prop_schema)
                if not is_optional:
                    google_required.append(prop_name)

            if google_required:
                google_schema["required"] = google_required

        # Convert array items
        if "items" in schema:
            google_schema["items"] = convert(schema["items"])

        return google_schema

    # Dereference if using $defs
    if "$defs" in json_schema:
        resolved_schema = jsonref.loads(json.dumps(json_schema))
        return convert(resolved_schema)
    else:
        return convert(json_schema)
