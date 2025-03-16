import logging
import os
from functools import lru_cache

import vertexai
from openai import OpenAI

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def init_vertaxai() -> None:
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    gcp_region = os.environ.get("GCP_REGION", "us-central1")
    logger.debug(f"Init Vertex AI with project {gcp_project_id} and region {gcp_region}")
    vertexai.init(project=gcp_project_id, location=gcp_region)


@lru_cache(maxsize=1)
def openai_client():
    return OpenAI()
