import base64
import json
import logging
import os
from functools import lru_cache

import google.auth
import requests
from dotenv import load_dotenv
from flask import Request
from google.cloud import logging as cloud_logging
from google.cloud import pubsub_v1
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

load_dotenv()


@lru_cache(maxsize=1)
def create_publisher():
    credentials = google_cloud_credentials(
        scopes=["https://www.googleapis.com/auth/pubsub"]
    )
    if credentials:
        return pubsub_v1.PublisherClient(credentials=credentials)
    else:
        return pubsub_v1.PublisherClient()  # Use default credentials


def google_cloud_credentials(scopes: list[str]) -> service_account.Credentials | None:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        return service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scopes
        )
    else:
        return None


def google_cloud_id_token(
    audience: str,
) -> str | None:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        credentials = service_account.IDTokenCredentials.from_service_account_file(
            credentials_path,
            target_audience=audience,
        )
        credentials.refresh(google.auth.transport.requests.Request())  # pyright: ignore
        return credentials.token
    else:
        return None


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
    for app_module in ["edgar_funcs", "main", "func_helpers"]:
        logging.getLogger(app_module).setLevel(app_log_level)


def get_default_project_id():
    _, project_id = google.auth.default()
    return project_id


def publish_message(message: dict, topic_name: str):
    gcp_proj_id = get_default_project_id()
    if gcp_proj_id and topic_name:
        publisher = create_publisher()
        topic_path = publisher.topic_path(gcp_proj_id, topic_name)

        content = json.dumps(message).encode("utf-8")
        future = publisher.publish(topic_path, content)
        message_id = future.result()  # Ensure the publish succeeds

        logger.debug(
            f"Published message ID {message_id} to {topic_name} with content {content}"
        )
    else:
        logging.info(f"Invalid topic {topic_name} or project {gcp_proj_id}")


def decode_request(request: Request) -> tuple[dict | None, dict | None]:
    """
    unwrap the http request sent by Pub/Sub push subscription
    return tuple of (headers, data) if successful, None otherwise
    """
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        return None, None

    pubsub_message = envelope["message"]
    decoded_data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
    data = json.loads(decoded_data)
    # headers are not needed for not, so  not implemented
    return {}, data


def send_cloud_run_request(url: str, payload: dict):
    """
    Sends a POST request to a Cloud Run HTTP endpoint with a JSON payload.
    Ensures the request is authenticated using default Google Cloud credentials.
    """
    auth_token = google_cloud_id_token(audience=url)
    if not auth_token:
        raise ValueError("No access token available after refreshing credentials.")

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    payload_bytes = json.dumps(payload).encode("utf-8")
    pubsub_like_payload = {
        "message": {"data": base64.b64encode(payload_bytes).decode("utf-8")},
    }
    response = requests.post(url, headers=headers, json=pubsub_like_payload)
    if response.status_code == 200:
        return response.json()
    raise ValueError(f"response status {response.status_code}\n{response.text}")
