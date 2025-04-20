import base64
import json
import logging
import os
from datetime import UTC, datetime, timedelta

import google.auth
import requests
from dotenv import load_dotenv
from flask import Request
from google.api_core.exceptions import NotFound
from google.cloud import firestore
from google.cloud import logging as cloud_logging
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


def insert_into_firestore(collection_name: str, data: dict):
    """
    Inserts a dictionary into a specified Firestore collection.
    """
    db = firestore.Client()
    collection_ref = db.collection(collection_name)
    collection_ref.add(data)
    logger.info(f"Inserted data into Firestore collection '{collection_name}': {data}")


def mark_job_in_progress(
    job_id: str, collection_name: str = "jobs_in_progress", ttl_hours: int = 24
) -> bool:
    """
    Marks a filing as in progress by inserting it into the specified Firestore collection.
    If a document with the same job_id already exists, it returns False.
    Otherwise, it inserts a new document and returns True.

    This is used as a de-duplication mechanism to prevent reprocessing the same filing,
    i.e. when pub/sub message is redelivered.

    Args:
        job_id (str): The unique identifier for the job.
        collection_name (str): The Firestore collection name.
        ttl_hours (int): Time-to-Live in hours for the document.
    """
    db = firestore.Client()
    doc_ref = db.collection(collection_name).document(job_id)

    @firestore.transactional
    def transaction_logic(transaction):
        try:
            doc_snapshot = doc_ref.get(transaction=transaction)
            if doc_snapshot.exists:
                return False
        except NotFound:
            pass

        transaction.set(
            doc_ref,
            {
                "id": job_id,
                "expires_at": datetime.now(UTC) + timedelta(hours=ttl_hours),
            },
        )
        return True

    transaction = db.transaction()
    return transaction_logic(transaction)


def mark_job_done(job_id: str, collection_name: str = "jobs_in_progress") -> bool:
    """
    Marks a job as done by deleting its document from the specified Firestore collection.
    If the document does not exist, it handles the exception gracefully.

    Args:
        job_id (str): The unique identifier for the job.
        collection_name (str): The Firestore collection name.

    Returns:
        bool: True if the document was deleted, False if it did not exist.
    """
    db = firestore.Client()
    doc_ref = db.collection(collection_name).document(job_id)

    try:
        doc_ref.delete()
        return True
    except NotFound:
        return False
