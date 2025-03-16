import logging
import os

from google.cloud import logging as cloud_logging

logger = logging.getLogger(__name__)


def setup_cloud_logging():
    # Only initialize Google Cloud Logging in Cloud Run
    if os.getenv("K_SERVICE"):
        client = cloud_logging.Client()
        client.setup_logging()
        logging.info("Google Cloud Logging is enabled.")
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info("Using local logging.")


if __name__ == "__main__":
    setup_cloud_logging()
