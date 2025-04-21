import sys

from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

load_dotenv()


def delete_collection(coll_ref, batch_id, batch_size=200):
    if batch_id:
        query = coll_ref.where(filter=FieldFilter("batch_id", "==", batch_id))
    else:
        query = coll_ref

    batch = coll_ref._client.batch()
    deleted = 0

    for doc in query.limit(batch_size).stream():
        batch.delete(doc.reference)
        deleted += 1

    if deleted > 0:
        batch.commit()  # Commit the batch
        print(f"Batch of {deleted} documents deleted.")
        if deleted >= batch_size:
            return delete_collection(coll_ref, batch_size)


def delete_document_and_subcollections(doc_ref):
    # Delete all subcollections first
    for subcollection in doc_ref.collections():
        for doc in subcollection.stream():
            delete_document_and_subcollections(doc.reference)
    # Delete the main document
    doc_ref.delete()
    print(f"Deleted: {doc_ref.path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zap.py <collection_name> <batch_id>")
        sys.exit(1)

    collection_name = sys.argv[1]
    batch_id = sys.argv[2] if len(sys.argv) > 2 else None

    db = firestore.Client()
    collection_ref = db.collection(collection_name)
    delete_collection(collection_ref, batch_id)
