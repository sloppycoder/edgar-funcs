import logging
import os
from typing import Any

import google.auth  # Add this import
from cloudevents.http import CloudEvent, to_json
from google.cloud import logging as cloud_logging
from google.cloud import pubsub_v1

MESSAGE_SOURCE = "edgar.funcs"


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
    for app_module in ["edgar", "rag", "main"]:
        logging.getLogger(app_module).setLevel(app_log_level)


def get_default_project_id():
    _, project_id = google.auth.default()
    return project_id


def publish_message(event: CloudEvent, topic_name: str):
    gcp_proj_id = get_default_project_id()
    if gcp_proj_id and topic_name:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(gcp_proj_id, topic_name)

        future = publisher.publish(topic_path, to_json(event))
        future.result()  # Ensure the publish succeeds
    else:
        logging.info(f"Invalid topic {topic_name} or project {gcp_proj_id}")


def publish_response(params: dict[str, Any], is_success: bool, msg: str):
    data = {
        "params": params,
        "success": is_success,
        "message": msg,
    }
    event = CloudEvent({}, data)
    publish_message(event, os.getenv("RESPONSE_TOPIC", ""))


def publish_request(data: dict[str, Any]):
    event = CloudEvent({}, data)
    publish_message(event, os.getenv("REQUEST_TOPIC", ""))
