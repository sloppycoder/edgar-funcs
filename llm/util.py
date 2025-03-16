import os
from functools import lru_cache

import vertexai
from openai import OpenAI


@lru_cache(maxsize=1)
def init_vertaxai() -> None:
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    gcp_region = os.environ.get("GCP_REGION", "us-central1")
    vertexai.init(project=gcp_project_id, location=gcp_region)


@lru_cache(maxsize=1)
def openai_client():
    return OpenAI()
