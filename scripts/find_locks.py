import json
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
from google.cloud import storage

from edgar_funcs.rag.vectorize import _storage_prefix

load_dotenv()


def find_locks(bucket_name: str, path: str, force_delete: bool = False):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=path)

    for blob in blobs:
        if blob.name.endswith("_lock.json"):
            try:
                content = json.loads(blob.download_as_text())
                expired_at = datetime.fromisoformat(content["expires_at"])
                now = datetime.now(timezone.utc)

                if expired_at < now:
                    print(f"Deleting expired lock: {blob.name}")
                    blob.delete()
                elif force_delete:
                    print(f"Force deleting lock: {blob.name}")
                    blob.delete()
                else:
                    print(f"Lock not expired: {blob.name}, expires at {expired_at}")
            except KeyError:
                print(f"Invalid lock file format: {blob.name}")
            except NotFound:
                pass


if __name__ == "__main__":
    force_delete = len(sys.argv) > 1 and sys.argv[1] == "-d"
    bucket_name, path = _storage_prefix("gs://edgar_666/edgar-funcs")
    if bucket_name:
        find_locks(bucket_name, path)
    else:
        print("STORAGE_PREFIX is not set to GCS")
