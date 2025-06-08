import logging
from functools import lru_cache

from google.cloud import storage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def gcs_client():
    return storage.Client()
