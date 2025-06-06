import json
import logging
from datetime import datetime
from typing import Optional, Type

import jsonref
import litellm
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..helper import init_vertaxai

logger = logging.getLogger(__name__)


def ask_model(
    model: str, prompt: str, responseModelClass: Type[BaseModel]
) -> Optional[str]:
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
    retry=retry_if_exception_type(Exception),
)
def _chat_with_litellm(
    model_name: str,
    prompt: str,
    responseModelClass: Type[BaseModel],
) -> Optional[str]:
    try:
        # Set up vertex AI for Gemini models
        if model_name.startswith("vertex_ai/"):
            init_vertaxai()

        # Prepare response format based on model type
        if model_name.startswith("vertex_ai/gemini"):
            # For Gemini models, use response_schema format
            google_schema = _convert_json_schema_to_google_schema(
                responseModelClass.model_json_schema()
            )
            response_format = {"type": "json_object", "response_schema": google_schema}
            max_tokens = 4096
        else:
            # For OpenAI models, use response_format
            response_format = {"type": "json_object"}
            max_tokens = 8192

        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            response_format=response_format,
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

    except Exception as e:
        logger.info(f"retrying {model_name} API call due to {type(e)}: {str(e)}")
        raise

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
