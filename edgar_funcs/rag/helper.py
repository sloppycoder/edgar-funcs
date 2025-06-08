from functools import lru_cache

from google.cloud import storage


@lru_cache(maxsize=1)
def gcs_client():
    return storage.Client()
