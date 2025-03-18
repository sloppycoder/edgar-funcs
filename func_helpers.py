import logging
import os
from typing import Any

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


def publish_message(event: CloudEvent, topic_name: str):
    gcp_proj_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if gcp_proj_id and topic_name:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(gcp_proj_id, topic_name)

        future = publisher.publish(topic_path, to_json(event))
        future.result()  # Ensure the publish succeeds
    else:
        logging.info(f"Invalid topic {topic_name} or project {gcp_proj_id}")


def publish_response(
    event_type: str, input_data: dict[str, Any], is_success: bool, msg: str
):
    attributes = {
        "type": event_type.replace(".req", ".resp"),
        "source": MESSAGE_SOURCE,
    }
    data = {
        "params": input_data,
        "success": is_success,
        "message": msg,
    }
    event = CloudEvent(attributes, data)
    publish_message(event, os.getenv("RESPONSE_TOPIC", ""))


def publish_request(event_type: str, input_data: dict[str, Any]):
    event = CloudEvent(
        {
            "type": event_type,
            "source": MESSAGE_SOURCE,
        },
        input_data,
    )
    publish_message(event, os.getenv("REQUEST_TOPIC", ""))
