import json
import logging
from typing import Optional

from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from vertexai.generative_models import GenerativeModel

from ..helper import init_vertaxai, openai_client

logger = logging.getLogger(__name__)


def ask_model(model: str, prompt: str) -> Optional[str]:
    if model.startswith("gemini"):
        return _chat_with_gemini(model, prompt)
    elif model.startswith("gpt"):
        return _chat_with_gpt(model, prompt)
    else:
        raise ValueError(f"Unknown model: {model}")


def remove_md_json_wrapper(response: str) -> str | None:
    # the response should be a JSON
    # sometimes Gemini wraps it in a markdown block ```json ...```
    # so we unrap the markdown block and get to the json
    if len(response) > 20:
        json_str = response.strip()
        start_markdown_index = response.find("```json")
        end_markdown__index = response.rfind("```")
        if start_markdown_index >= 0 and end_markdown__index >= 0:
            json_str = response[start_markdown_index + 7 : end_markdown__index]

        try:
            json.loads(json_str)  # just to test if the json is valid
            return json_str
        except json.JSONDecodeError:
            logger.info("Failed to parse JSON from response")

    return response


def _chat_with_gpt(model_name: str, prompt: str) -> Optional[str]:
    client = openai_client()

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.warning(f"Error calling OpenAI API: {type(e)},{str(e)}")
        return None


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    retry=retry_if_exception_type(ResourceExhausted),
)
def _chat_with_gemini(model_name: str, prompt: str) -> Optional[str]:
    try:
        init_vertaxai()
        model = GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "temperature": 0,
                "top_p": 0.95,
            },
        )
        return response.text
    except ResourceExhausted:
        # for tenacity to retry
        raise
    except Exception as e:
        logging.warning(f"Error calling Gemini API: {type(e)},{str(e)}")
        return None
