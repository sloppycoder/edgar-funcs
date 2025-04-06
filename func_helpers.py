import json
import logging
import os
from typing import Any

import google.auth  # Add this import
from dotenv import load_dotenv
from google.cloud import logging as cloud_logging
from google.cloud import pubsub_v1
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

load_dotenv()


def google_cloud_credentials(scopes: list[str]) -> service_account.Credentials | None:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        return service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scopes
        )
    else:
        return None


def model_settings():
    extraction_model = os.environ.get("EXTRACTION_MODEL")
    if not extraction_model:
        raise RuntimeError("EXTRACTION_MODEL not set")

    embedding_model = os.environ.get("EMBEDDING_MODEL")
    if not embedding_model:
        raise RuntimeError("EMBEDDING_MODEL not set")

    dimension = os.environ.get("EMBEDDING_DIMENSION", "0")
    try:
        embedding_dimension = int(dimension)
        return embedding_model, embedding_dimension, extraction_model
    except ValueError:
        raise RuntimeError("EMBEDDING_DIMENSION must be set to a integer value")


def setup_cloud_logging():
    # Only initialize Google Cloud Logging in Cloud Run
    if os.getenv("K_SERVICE"):
        client = cloud_logging.Client()
        client.setup_logging()
        logging.info("Google Cloud Logging is enabled.")
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info("Using local logging.")

    app_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO"), logging.INFO)
    for app_module in ["edgar", "rag", "main", "func_helpers"]:
        logging.getLogger(app_module).setLevel(app_log_level)


def get_default_project_id():
    _, project_id = google.auth.default()
    return project_id


def publish_message(message: dict, topic_name: str):
    gcp_proj_id = get_default_project_id()
    if gcp_proj_id and topic_name:
        credentials = google_cloud_credentials(
            scopes=["https://www.googleapis.com/auth/pubsub"]
        )
        if credentials:
            publisher = pubsub_v1.PublisherClient(credentials=credentials)
        else:
            publisher = pubsub_v1.PublisherClient()  # Use default credentials

        topic_path = publisher.topic_path(gcp_proj_id, topic_name)

        content = json.dumps(message).encode("utf-8")
        future = publisher.publish(topic_path, content)
        message_id = future.result()  # Ensure the publish succeeds

        logger.debug(
            f"Published message ID {message_id} to {topic_name} with content {content}"
        )
    else:
        logging.info(f"Invalid topic {topic_name} or project {gcp_proj_id}")


def publish_response(params: dict[str, Any], is_success: bool, msg: str):
    data = {
        "params": params,
        "success": is_success,
        "message": msg,
    }
    publish_message(data, os.getenv("RESPONSE_TOPIC", ""))


def publish_request(data: dict[str, Any]):
    publish_message(data, os.getenv("REQUEST_TOPIC", ""))
